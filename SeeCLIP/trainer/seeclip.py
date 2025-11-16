import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torchvision.transforms as transforms
import os
from torchvision.models import resnet18, resnet50
from torchvision.transforms import ToTensor

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from diffusers import StableDiffusionPipeline
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = _Tokenizer()
df = pd.DataFrame()
cls_label = pd.DataFrame()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class SemanticTokenExtractor(nn.Module):
    """Semantic token extractor - Implements K-head attention mechanism for fine-grained semantic extraction"""
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # K learnable query vectors
        self.query_vectors = nn.Parameter(torch.randn(num_heads, d_model))
        nn.init.normal_(self.query_vectors, std=0.02)

    def forward(self, patch_embeddings):
        """
        Args:
            patch_embeddings: [batch_size, num_patches, d_model]
        Returns:
            semantic_tokens: [batch_size, num_heads, d_model]
        """
        batch_size, num_patches, d_model = patch_embeddings.shape
        semantic_tokens = []
        
        for k in range(self.num_heads):
            # Calculate attention scores (Equation 1)
            attention_scores = torch.matmul(patch_embeddings, self.query_vectors[k])  # [batch_size, num_patches]
            attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_patches]
            
            # Weighted aggregation to get the k-th semantic token (Equation 2)
            semantic_token = torch.sum(
                attention_weights.unsqueeze(-1) * patch_embeddings, dim=1
            )  # [batch_size, d_model]
            semantic_tokens.append(semantic_token)
        
        return torch.stack(semantic_tokens, dim=1)  # [batch_size, num_heads, d_model]

class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(4, 3, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UpsampleNetwork(nn.Module):
    def __init__(self):
        super(UpsampleNetwork, self).__init__()
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.conv4 = nn.ConvTranspose2d(64, 1, kernel_size=7, stride=3, padding=1, output_padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x.unsqueeze(-1).unsqueeze(-1)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class DomainTokenGenerator(nn.Module):
    """Domain token generator"""
    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model

    def forward(self, features):
        """
        Args:
            features: [batch_size, d_model] Image features
        Returns:
            domain_token: [batch_size, d_model] Domain token
        """
        # Simple average as domain token
        if len(features.shape) == 3:  # If it's patch features
            domain_token = torch.mean(features, dim=1)
        else:
            domain_token = features
        return domain_token


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts.to(device) + self.positional_embedding.type(self.dtype).to(device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class SemanticAwarePromptLearner(nn.Module):
    """Semantic-aware prompt learner - Implements enhanced prompt construction"""
    def __init__(self, classnames, clip_model, num_semantic_heads=8, unknown_tokens=4):
        super().__init__()
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.num_semantic_heads = num_semantic_heads
        self.unknown_tokens = unknown_tokens
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # Semantic token extractor
        self.semantic_extractor = SemanticTokenExtractor(ctx_dim, num_semantic_heads)
        
        # Domain token generator
        self.domain_generator = DomainTokenGenerator(ctx_dim)
        
        # Projection functions Φ and Ψ
        self.domain_projection = nn.Linear(ctx_dim, ctx_dim)  # Φ function
        self.semantic_projections = nn.ModuleList([
            nn.Linear(ctx_dim, ctx_dim) for _ in range(num_semantic_heads)  # Ψ function
        ])
        
        # Learnable semantic vectors for unknown categories
        self.unknown_semantic_tokens = nn.Parameter(torch.randn(unknown_tokens, ctx_dim, dtype=dtype))
        nn.init.normal_(self.unknown_semantic_tokens, std=0.02)
        
        # Prepare class name embeddings
        classnames = [name.replace("_", " ") for name in classnames]
        
        # Build known class prompt templates
        known_prompts = [f"a photo of a {name}." for name in classnames]
        self.known_tokenized_prompts = torch.cat([clip.tokenize(p) for p in known_prompts])
        
        # Build unknown class prompt template
        unknown_prompt = "a photo of an unknown object."
        self.unknown_tokenized_prompt = clip.tokenize([unknown_prompt])
        
        with torch.no_grad():
            # Get known class embeddings
            known_embedding = clip_model.token_embedding(self.known_tokenized_prompts).type(dtype)
            self.register_buffer("known_token_prefix", known_embedding[:, :1, :])
            self.register_buffer("known_token_suffix", known_embedding[:, -1:, :])
            
            # Get unknown class embeddings
            unknown_embedding = clip_model.token_embedding(self.unknown_tokenized_prompt).type(dtype)
            self.register_buffer("unknown_token_prefix", unknown_embedding[:, :1, :])
            self.register_buffer("unknown_token_suffix", unknown_embedding[:, -1:, :])

    def construct_known_prompts(self, domain_tokens, semantic_tokens):
        """Construct enhanced prompts for known classes (Equation 3)"""
        batch_size = domain_tokens.shape[0]
        
        # Project domain tokens and semantic tokens
        projected_domain = self.domain_projection(domain_tokens)  # [batch_size, ctx_dim]
        projected_semantics = []
        for k in range(self.num_semantic_heads):
            projected_sem = self.semantic_projections[k](semantic_tokens[:, k, :])  # [batch_size, ctx_dim]
            projected_semantics.append(projected_sem)
        projected_semantics = torch.stack(projected_semantics, dim=1)  # [batch_size, num_heads, ctx_dim]
        
        # Construct prompt sequences
        prompts = []
        for b in range(batch_size):
            for c in range(self.num_classes):
                # [prefix] + [domain] + [semantic_1, ..., semantic_K] + [suffix]
                prompt_tokens = torch.cat([
                    self.known_token_prefix[c],  # [1, ctx_dim]
                    projected_domain[b:b+1],     # [1, ctx_dim]
                    projected_semantics[b],      # [num_heads, ctx_dim]
                    self.known_token_suffix[c]   # [1, ctx_dim]
                ], dim=0)  # [1 + 1 + num_heads + 1, ctx_dim]
                prompts.append(prompt_tokens)
        
        return torch.stack(prompts)  # [batch_size * num_classes, seq_len, ctx_dim]

    def construct_unknown_prompts(self, domain_tokens):
        """Construct prompts for unknown classes (Equation 4)"""
        batch_size = domain_tokens.shape[0]
        
        # Project domain tokens
        projected_domain = self.domain_projection(domain_tokens)  # [batch_size, ctx_dim]
        
        # Construct unknown prompt sequence
        prompts = []
        for b in range(batch_size):
            # [prefix] + [domain] + [v_1, ..., v_m] + [unknown] + [suffix]
            prompt_tokens = torch.cat([
                self.unknown_token_prefix[0],    # [1, ctx_dim]
                projected_domain[b:b+1],         # [1, ctx_dim]
                self.unknown_semantic_tokens,    # [unknown_tokens, ctx_dim]
                self.unknown_token_suffix[0]     # [1, ctx_dim]
            ], dim=0)  # [1 + 1 + unknown_tokens + 1, ctx_dim]
            prompts.append(prompt_tokens)
        
        return torch.stack(prompts)  # [batch_size, seq_len, ctx_dim]

    def forward(self, patch_embeddings, global_features):
        """
        Args:
            patch_embeddings: [batch_size, num_patches, d_model]
            global_features: [batch_size, d_model]
        Returns:
            known_prompts, unknown_prompts, semantic_tokens
        """
        # Extract semantic tokens
        semantic_tokens = self.semantic_extractor(patch_embeddings)
        
        # Generate domain tokens
        domain_tokens = self.domain_generator(global_features)
        
        # Construct prompts
        known_prompts = self.construct_known_prompts(domain_tokens, semantic_tokens)
        unknown_prompts = self.construct_unknown_prompts(domain_tokens)
        
        return known_prompts, unknown_prompts, semantic_tokens


class SemanticGuidedDiffusion(nn.Module):
    """Semantic-guided diffusion module"""
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        self.model_id_or_path = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id_or_path, 
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(device)
        logging.basicConfig(level=logging.WARNING)

    def perturb_semantic_tokens(self, semantic_tokens):
        """Add Gaussian noise to semantic tokens (Equation 8)"""
        noise = torch.normal(0, self.sigma, size=semantic_tokens.shape).to(semantic_tokens.device)
        return semantic_tokens + noise

    def generate_pseudo_unknowns(self, semantic_tokens, domain_name, known_classes, batch_size=4):
        """Generate pseudo-unknown samples"""
        # Perturb semantic tokens
        perturbed_tokens = self.perturb_semantic_tokens(semantic_tokens)
        
        # Build positive prompt
        positive_prompt = f"A {domain_name} image of an unknown class"
        
        # Build negative prompt
        negative_prompt = ", ".join(known_classes)
        
        # Generate images
        generated_images = []
        actual_batch_size = min(batch_size, max(1, int(semantic_tokens.shape[0] * 0.1)))
        
        with torch.no_grad():
            for i in range(actual_batch_size):
                batch_output = self.pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=15,
                    num_inference_steps=20
                )
                generated_images.append(batch_output.images[0])
        
        # Convert to tensor and preprocess
        generated_images = torch.stack([ToTensor()(img) for img in generated_images]).to(device)
        
        # Resize and normalize
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        normalize = transforms.Normalize(mean=mean, std=std)
        
        resized_images = torch.stack([resize_transform(x) for x in generated_images])
        normalized_images = normalize(resized_images).to(device)
        
        return normalized_images


class SeeCLIP(nn.Module):
    """SeeCLIP main model"""
    def __init__(self, classnames, domainnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.domainnames = domainnames
        self.num_classes = len(classnames)
        self.dtype = clip_model.dtype
        
        # Core components
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        
        # Semantic-aware prompt learner
        self.prompt_learner = SemanticAwarePromptLearner(classnames, clip_model)
        
        # Semantic-guided diffusion module
        self.diffusion_generator = SemanticGuidedDiffusion()

    def compute_repulsion_loss(self, unknown_text_features, known_text_features, margin=0.5):
        """Compute repulsion loss (Equation 5)"""
        repulsion_loss = 0
        for known_feat in known_text_features:
            similarity = F.cosine_similarity(unknown_text_features, known_feat, dim=-1)
            repulsion_loss += torch.sum(torch.clamp(margin - similarity, min=0))
        return repulsion_loss / len(known_text_features)

    def compute_cohesive_loss(self, unknown_text_features, known_text_features):
        """Compute cohesion loss (Equation 6)"""
        center = torch.mean(torch.stack(known_text_features), dim=0)
        return F.mse_loss(unknown_text_features, center)

    def compute_regularization_loss(self, semantic_projections, lambda_reg=0.01):
        """Compute regularization loss (Equation 7)"""
        reg_loss = 0
        for proj in self.prompt_learner.semantic_projections:
            for param in proj.parameters():
                reg_loss += torch.norm(param, p=1)
        return lambda_reg * reg_loss

    def forward(self, image, labels=None, generate_unknowns=False):
        """
        Args:
            image: [batch_size, 3, 224, 224]
            labels: [batch_size] Optional labels
            generate_unknowns: Whether to generate pseudo-unknown samples
        """
        batch_size = image.shape[0]
        
        # Extract image features
        global_features, patch_embeddings = self.image_encoder(image.type(self.dtype))
        global_features = global_features / global_features.norm(dim=-1, keepdim=True)
        
        # Construct enhanced prompts
        known_prompts, unknown_prompts, semantic_tokens = self.prompt_learner(
            patch_embeddings, global_features
        )
        
        # Compute text features
        # Reshape known_prompts to fit the text_encoder
        known_prompts_reshaped = known_prompts.view(-1, known_prompts.shape[-2], known_prompts.shape[-1])
        known_tokenized = self.prompt_learner.known_tokenized_prompts.repeat(batch_size, 1).to(device)
        
        known_text_features = self.text_encoder(known_prompts_reshaped, known_tokenized)
        known_text_features = known_text_features / known_text_features.norm(dim=-1, keepdim=True)
        known_text_features = known_text_features.view(batch_size, self.num_classes, -1)
        
        # Unknown class text features
        unknown_tokenized = self.prompt_learner.unknown_tokenized_prompt.repeat(batch_size, 1).to(device)
        unknown_text_features = self.text_encoder(unknown_prompts, unknown_tokenized)
        unknown_text_features = unknown_text_features / unknown_text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity scores
        logit_scale = self.logit_scale.exp()
        
        # Known class logits
        known_logits = []
        for b in range(batch_size):
            logits_b = logit_scale * global_features[b] @ known_text_features[b].t()
            known_logits.append(logits_b)
        known_logits = torch.stack(known_logits)
        
        # Unknown class logits
        unknown_logits = logit_scale * torch.sum(global_features * unknown_text_features, dim=-1, keepdim=True)
        
        # Combine logits
        all_logits = torch.cat([known_logits, unknown_logits], dim=-1)
        
        # Compute loss
        losses = {}
        
        if labels is not None:
            # Alignment loss (Equation 10)
            alignment_loss = F.cross_entropy(known_logits, labels)
            losses['alignment'] = alignment_loss
            
            # Repulsion loss (Equation 5)
            repulsion_loss = self.compute_repulsion_loss(
                unknown_text_features.mean(dim=0), 
                [known_text_features[b, labels[b]] for b in range(batch_size)]
            )
            losses['repulsion'] = repulsion_loss
            
            # Cohesion loss (Equation 6)
            cohesive_loss = self.compute_cohesive_loss(
                unknown_text_features.mean(dim=0),
                [known_text_features[b, labels[b]] for b in range(batch_size)]
            )
            losses['cohesive'] = cohesive_loss
            
            # Regularization loss (Equation 7)
            reg_loss = self.compute_regularization_loss()
            losses['regularization'] = reg_loss
        
        results = {
            'logits': all_logits,
            'known_logits': known_logits,
            'unknown_logits': unknown_logits,
            'semantic_tokens': semantic_tokens,
            'losses': losses
        }
        # Generate pseudo-unknown samples (if needed)
        if generate_unknowns and len(self.domainnames) > 0:
            domain_name = self.domainnames[0] if self.domainnames else "photo"
            pseudo_unknowns = self.diffusion_generator.generate_pseudo_unknowns(
                semantic_tokens, domain_name, self.classnames
            )
            results['pseudo_unknowns'] = pseudo_unknowns
        return results


