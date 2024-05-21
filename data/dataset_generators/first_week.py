from .base import DatasetGenerator
import json


class FirstWeekGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label

    def create_prompts(self, labels_names):
        prompts = {}

        with open("/X/MODAL/Challenge/challenge-modal/prompts.json", "r") as file:
            prompts_data = json.load(file)

        for label, data in prompts_data.items():
            prompts[label] = []
            prompts[label].append(
            {
                "prompt": data["prompt"],
                "num_images": self.num_images_per_label,
            }
        )

        return prompts