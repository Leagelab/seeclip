# The Finer the Better: Towards Granular-aware Open-set Domain Generalization

## Project Overview
This project introduces a novel approach to **Open-Set Domain Generalization (OSDG)**, a critical problem in machine learning that involves recognizing both known and unknown classes across different domains. The goal of this research is to develop a more robust and precise method for distinguishing between known and unknown categories, especially in situations where unseen, open classes may appear.

To address the challenges of OSDG, we propose **Semantic-enhanced CLIP (SeeCLIP)**, a framework that leverages fine-grained semantics to improve unknown sample detection. SeeCLIP integrates fine-grained semantic features into vision-language models, specifically utilizing CLIP (Contrastive Language-Image Pre-training), to enhance the model's generalization ability in open-set environments.

## Key Contributions
1. **SeeCLIP Framework**: SeeCLIP enhances CLIP by incorporating a **semantic-aware prompt enhancement module** and a **semantic-guided diffusion module**, enabling the model to more accurately identify unknown classes by focusing on subtle semantic differences.
   
2. **Duplex Contrastive Learning**: This approach enables the model to learn prompts that are both similar to known categories but also exhibit key semantic differences, which helps the model avoid overfitting to the known classes.

3. **Pseudo-Open Generation**: The framework also introduces a method for generating **pseudo-open** samples, which are designed to closely resemble unknown classes but maintain clear distinctions from known ones.

4. **Generalization Bound**: A theoretical bound is derived for OSDG, showing that SeeCLIP effectively reduces the generalization risk by balancing the structural risk of known classes and the open space risk of unknown classes.

5. **State-of-the-Art Performance**: Extensive experiments across benchmark datasets demonstrate that SeeCLIP outperforms existing methods by improving accuracy and H-index, with a significant increase in robustness for open-set recognition.

## Main Features
- **Fine-grained Semantic Modeling**: SeeCLIP enhances vision-language models by incorporating fine-grained semantic tokens to capture discriminative features from different parts of the image.
- **Robust Open-set Recognition**: By addressing both structural and open space risks, SeeCLIP offers superior performance in distinguishing unknown classes, even in complex, real-world scenarios.
- **Efficient Unknown Generation**: The **semantic-guided diffusion module** generates pseudo-open samples, ensuring better generalization and reducing biases in classifying unknowns.
  
## Model Architecture
- **Semantic-aware Prompt Enhancement**: Extracts fine-grained semantic features from images and integrates them into text prompts for more precise vision-language alignment.
- **Diffusion Model for Pseudo-Open Generation**: Uses a pre-trained diffusion model to synthesize pseudo-unknown samples that are globally similar to known classes but exhibit local semantic differences.
- **Duplex Contrastive Learning**: Optimizes both prompt learning and pseudo-open generation to ensure clear boundaries between known and unknown classes.

## Experimental Results
SeeCLIP has been tested on several benchmark datasets, including **Office-Home**, **PACS**, and **Mini-DomainNet**, and has consistently outperformed state-of-the-art methods, with notable improvements in both accuracy and H-score (harmonic mean of known and unknown accuracies).

## Conclusion
The SeeCLIP framework provides a comprehensive solution to the OSDG problem by utilizing semantic features and innovative learning techniques to effectively address the challenges of recognizing unknown classes in unseen domains.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
