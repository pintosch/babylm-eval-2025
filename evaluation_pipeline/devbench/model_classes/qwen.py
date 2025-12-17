from tqdm import tqdm
import torch
import numpy as np


class QwenEvalModel: 
    def __init__(self, model, processor=None, device="cpu"):
        self.device = device
        self.model = model.to(device).eval()
        self.processor = processor

    def get_all_image_feats(self, dataloader):
        """
        Gets image features from a dataloader
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        Outputs:
        - a numpy array of shape [num_images, embed_dim]
        """
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing images"):
                feats = self._get_image_features(d["images"])
                all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)

    def _get_image_features(self, images) -> np.ndarray:
        """
        Extract image features using the vision encoder.
        """
        # Process images with image_token as text prompt
        prompt_text = self.processor.image_token
        
        inputs = self.processor(
            images=images,
            text=prompt_text,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        
        visual_model = self.model.model.visual
        
        pixel_values = pixel_values.type(visual_model.dtype)
        image_embeds, _ = visual_model(pixel_values, grid_thw=image_grid_thw)
        
        # Split embeddings per image and mean pool each
        split_sizes = (image_grid_thw.prod(dim=-1) // visual_model.spatial_merge_size ** 2).tolist()
        image_embeds_split = torch.split(image_embeds, split_sizes, dim=0)
        
        feats = []
        for embed in image_embeds_split:
            feat = embed.mean(dim=0)
            feats.append(feat)
        
        feats = torch.stack(feats, dim=0)
        return feats.cpu().numpy()

    def get_all_text_feats(self, dataloader):
        """
        Gets text features from a dataloader
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        Outputs:
        - a numpy array of shape [num_texts, embed_dim]
        """
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing texts"):
                feats = self._get_text_features(d["text"])
                all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)

    def _get_text_features(self, texts) -> np.ndarray:
        """
        Extract text features using the language model's embeddings.
        """
        inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        embed_tokens = self.model.model.language_model.embed_tokens
        embeddings = embed_tokens(inputs.input_ids)
        
        # Mean pool over sequence (masked by attention)
        attention_mask = inputs.attention_mask.unsqueeze(-1)
        masked_embeddings = embeddings * attention_mask
        feats = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        
        return feats.cpu().numpy()

    def get_all_sim_scores(self, dataloader):
        """
        Gets image--text similarity scores from a dataloader
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        Outputs: 
        - a numpy array of shape [num_trials, num_images_per_trial, num_texts_per_trial]
        """
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing similarities"):
                sims = self._get_similarity_scores(d["images"], d["text"])
                all_sims.append(sims)
        return np.stack(all_sims, axis=0)

    def _get_similarity_scores(self, images, texts) -> np.ndarray:
        """
        Compute similarity scores between images and texts.
        Returns shape [num_images, num_texts]
        """
        num_images = len(images)
        num_texts = len(texts)
        
        scores = np.zeros((num_images, num_texts))
        
        for i, image in enumerate(images):
            for j, text in enumerate(texts):
                score = self._compute_image_text_score(image, text)
                scores[i, j] = score
        
        return scores

    def _compute_image_text_score(self, image, text:  str) -> float:
        """
        Compute a similarity score between an image and text using
        the model's log-likelihood of generating the text given the image.
        """
        prompt_text = f"{self.processor.image_token}{text}"
        
        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
        ).to(self.device)
        
        # Create labels: mask image tokens, only compute loss on text tokens
        labels = inputs.input_ids.clone()
        labels[inputs.input_ids == self.processor.image_token_id] = -100
        
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            labels=labels,
        )
        
        # Return negative loss as score (higher is better)
        return -outputs.loss.item()