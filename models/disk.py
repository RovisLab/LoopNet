import torch
import kornia.feature

class DISK:
    default_conf = {
        "weights": "depth",
        "max_num_keypoints": None,
        "desc_dim": 128,
        "nms_window_size": 5,
        "detection_threshold": 0.0,
        "pad_if_not_divisible": True,
    }

    preprocess_conf = {
        "resize": 1024,
        "grayscale": False,
    }

    required_data_keys = ["image"]

    def __init__(self, **conf) -> None:
        self.conf = {**self.default_conf, **conf}
        self.model = kornia.feature.DISK.from_pretrained(self.conf["weights"])
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
    def to(self, device):
        self.model = self.model.to(device)
        return self

    def forward(self, image: torch.Tensor) -> dict:
        device = image.device
        self.model = self.model.to(device)
        
        if len(image.shape) == 4 and image.shape[1] == 1:
            image = kornia.color.grayscale_to_rgb(image)
            
        features = self.model(
            image,
            n=self.conf["max_num_keypoints"],
            window_size=self.conf["nms_window_size"],
            score_threshold=self.conf["detection_threshold"],
            pad_if_not_divisible=self.conf["pad_if_not_divisible"],
        )
        
        batch_size = len(features)
        processed_descriptors = []
        
        for i in range(batch_size):
            curr_descriptors = features[i].descriptors
            avg_descriptor = torch.mean(curr_descriptors, dim=0)
            processed_descriptors.append(avg_descriptor)
        
        descriptors = torch.stack(processed_descriptors, dim=0)
        return {"descriptors": descriptors.to(device)}
