from evaluation_pipeline.devbench.eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np


class QwenEvalModel(EvalModel):
    def __init__(self, model, processor=None, device="cpu"):
        self.device = device
        self.model = model.to(device).eval()
        self.processor = processor
        # Set compatibility attributes for base class interface
        self.get_image_features = self.get_all_image_feats
        self.get_text_features = self.get_all_text_feats
        self.get_similarity_scores = self.get_all_sim_scores

    def get_all_sim_scores(self, dataloader):
        """
        Gets image-text similarity scores from a dataloader using Qwen model
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        Outputs:
        - a numpy array of shape [num_trials, num_images_per_trial, num_texts_per_trial]
        """
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                num_images = len(d["images"])
                num_texts = len(d["text"])
                sims = np.zeros((num_images, num_texts))

                for i, image in enumerate(d["images"]):
                    # Convert image to RGB if it's not already in that format
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    for j, text in enumerate(d["text"]):
                        # Format prompt similar to BabyLM models
                        prompt = f"The caption for this image is: {text}."
                        prompt_with_image = f"{self.processor.image_token}{prompt}"

                        # Process inputs
                        inputs = self.processor(
                            images=image,
                            text=prompt_with_image,
                            return_tensors="pt",
                            truncation=False,
                        ).to(self.device)

                        # Create labels: mask image tokens, only compute loss on text tokens
                        labels = inputs.input_ids.clone()
                        labels[inputs.input_ids == self.processor.image_token_id] = -100

                        # Forward pass
                        outputs = self.model(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            pixel_values=inputs.get("pixel_values"),
                            image_grid_thw=inputs.get("image_grid_thw"),
                            labels=labels,
                        )

                        # Use negative loss as similarity score (higher is better)
                        sims[i, j] = -outputs.loss.detach().cpu().numpy()

                all_sims.append(sims)

        return np.stack(all_sims, axis=0)

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
        batch_size = 4
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                images_rgb = [
                    image.convert("RGB") if image.mode != "RGB" else image
                    for image in d["images"]
                ]
                # Use a minimal prompt to get hidden states
                minimal_prompts = [self.processor.image_token + " "] * len(images_rgb)

                for start in range(0, len(images_rgb), batch_size):
                    end = start + batch_size
                    batch_images = images_rgb[start:end]
                    batch_prompts = minimal_prompts[start:end]

                    inputs = self.processor(
                        images=batch_images,
                        text=batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                    ).to(self.device)

                    labels = inputs.input_ids.clone()
                    labels[inputs.input_ids == self.processor.image_token_id] = -100

                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                        output_hidden_states=True,
                    )

                    # Mean pool the last hidden state
                    hidden_states = outputs.hidden_states[-1]
                    mean_feats = hidden_states.mean(dim=1).detach().cpu().numpy()
                    all_feats.append(mean_feats)

        return np.concatenate(all_feats, axis=0)

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
            for d in tqdm(dataloader, desc="Processing data"):
                # Extract text features
                texts = d["text"]

                inputs = self.processor.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ).to(self.device)

                outputs = self.model.model.language_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                )

                # Mean pool the last hidden state
                hidden_states = outputs.hidden_states[-1]
                mean_feats = hidden_states.mean(dim=1).detach().cpu().numpy()
                all_feats.append(mean_feats)

        return np.concatenate(all_feats, axis=0)
