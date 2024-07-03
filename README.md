# VIKCSE
Implementation of paper: 'VIKCSE: VIsual-Knowledge enhenced Constrastive Sentence Embedding'

## Abstract
Learning high-quality text representations is a fundamental task in natural language processing. Existing methods attempt to enhance sentence representation by incorporating more external non-linguistic knowledge, thereby alleviating the anisotropy of output sentence representations. However, due to the disparity in modalities, integrating visual knowledge into text requires specific methods to seamlessly fuse visual and textual features, which presents new challenges. In this paper, we propose the **Vi**sual-**K**nowledge enhanced **C**ontrastive learning with prompts for **S**entence **E**mbedding (**VIKCSE**) method to investigate how pre-trained language models can more efficiently integrate cross-modal external semantic knowledge. Initially, the method utilizes a pre-trained multimodal contrastive model to extract visual semantic features from publicly available image-text datasets, thereby constructing a visual knowledge base. Then, we proposed a prompt-based text contrastive learning model, which uses designed templates to generate sentence semantic embedding. It then retrieves matching visual features from the visual knowledge base, together with the original guided semantic sentence embedding, and participates in contrastive loss computing. We take BERT and Roberta, the two fundamental models, as examples and evaluate the performance of VIKCSE on both the Semantic Textual Similarity (STS) task and downstream transfer tasks. Compared to similar existing work, VIKCSE achieves an average Spearman correlation improvement of 2.79\% on STS tasks, demonstrating superior performance. 

## Highlights

1. We present a two-stage method incorporating visual knowledge into pretrained language models.
2. We explore how different prompt templates affect guided semantic sentence embedding.
3. The proposed method `borrowing' the image-text matching signals to text contrastive learning.
4. We carefully evaluate the model and each component in the ablation study and prove its effectiveness.

## Method
<center>Fig.1 Schematic diagram of the VIKCSE method.</center>

![这是图片](/img/Fig_model_structure_2.png "Schematic diagram of the VIKCSE method.")



Briefly, the method is divided into two independent stages. The first stage involves extracting visual features from a publicly available multimodal dataset using a pre-trained visual-textual model, culminating in constructing a visual semantic knowledge base. Recent studies have demonstrated that pre-trained models have successfully fused much knowledge about the multimodal world, leading to similar conclusions in text and information extraction. Drawing inspiration from these findings, we regard knowledge extraction from pre-trained visual models as a form of knowledge base, serving to supervise textual models and probe the supervisory potential of cross-modal information extraction.

The second phase employs prompt learning to extract [MASK] and [CLS] token features and retrieve the most similar image features within the visual semantic knowledge base. The rationale behind seeking the most similar features lies in previous research demonstrating that difficult negative samples with high similarity help in the training of contrastive models, thereby assisting models in discerning subtle differences between categories. Finally, the retrieved features also participate in the contrastive loss computation. In contrast to previous work, our proposed method eschews the insertion or splicing of visual features into textual representations. Instead, it harnesses contrastive learning to attain alignment properties of pre-trained text-visual features, optimizing its learning of text features.


## Main Results

<center>Fig.2 Schematic diagram of the VIKCSE method.</center>

![这是图片](/img/bert_results.png "Schematic diagram of the VIKCSE method.")

<center>Fig.3 Schematic diagram of the VIKCSE method.</center>

![这是图片](/img/Roberta_results.png "Schematic diagram of the VIKCSE method.")
