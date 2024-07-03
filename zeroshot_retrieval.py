import torch
import torch.nn.functional as F
from tqdm import tqdm

def zeroshot_evaluate(model, dataloader, processor, max_seq_length, device=torch.device("cuda"), recall_k_list=[5]):
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    inds = 0

    for i, data in tqdm(enumerate(dataloader)):
        # text = [" ".join('%s' %a for a in sentence) for sentence in caption]
        # image = [img.convert("RGB") for img in image]
        processed_data = processor(
            images=data['image'].convert("RGB"), 
            text=data['caption'],
            padding='max_length',
            max_length=77,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        batch_texts_image_index = inds+torch.arange(len(data['caption']))
        inds += len(data['caption'])
        # compute the embedding of images and texts
        with torch.no_grad():
            outputs = model(pixel_values=processed_data['pixel_values'].cuda(), 
                        input_ids=processed_data['input_ids'].cuda(),
                        attention_mask=processed_data['attention_mask'].cuda(),
                        output_hidden_states=True,
                        )
            batch_images_emb = F.normalize(outputs['image_embeds'])
            batch_texts_emb = F.normalize(outputs['text_embeds'])
            # batch_images_emb = F.normalize(model.get_image_features(processed_data['pixel_values'].cuda()), dim=-1)
            # batch_texts_emb = F.normalize(model.get_text_features(input_ids=processed_data['input_ids'].cuda(),attention_mask=processed_data['attention_mask'].cuda()), dim=-1)

        batch_images_emb_list.append(batch_images_emb)
        batch_texts_emb_list.append(batch_texts_emb)
        texts_image_index.extend([batch_texts_image_index])
        
    batch_size = len(batch_images_emb_list[0])  # 1
    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)  
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores  = texts_emb @ images_emb.t()    # torch.Size([5070, 1014])
    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool).cpu()  # torch.Size([5070, 1014])

    for idx, val in enumerate(texts_image_index):
        positive_pairs[val.tolist(), idx] = True

    metrics = {}
    for recall_k in recall_k_list:

        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs.cuda(), batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.cuda().T, batch_size, device, k=recall_k)>0).float().mean().item()

    return metrics


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end]
        y = Y[start:end]
        result = func(x, y, *args, **kwargs)
        results.append(result)
    return torch.cat(results)
