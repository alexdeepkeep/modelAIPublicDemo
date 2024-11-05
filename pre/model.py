from torchvision.transforms import Resize
import torch


class Preprocessor:

    @staticmethod
    def execute(input_data: torch.Tensor) -> torch.Tensor:
        """
        This method should be implemented in the child class. It should take the input data,
        preprocess it and return the preprocessed data.

        :param input_data: The data to be preprocessed.
        :return: The preprocessed data.
        """
        assert (isinstance(input_data, list) and isinstance(input_data[0], torch.Tensor)
                and len(input_data[0].shape) == 4), ("input_data must be a torch.Tensor in the format BCHW, "
                                                     f"got {type(input_data[0])} with shape {input_data[0].shape}")
        images = []
        for image in input_data:
            images.append(Resize((416, 416))(image))
        return torch.cat(images)

