def saliency_map(self, grads, filenames, save_dir='./saliency_map/'):
    assert isinstance(grads, torch.Tensor)
    assert grads.dim() == 4 and grads.shape[1] == 3

    save_dir = os.path.join(save_dir, self.model_name, self.attack.lower().replace('-', ''), 'clean')
    os.makedirs(save_dir, exist_ok=True)
    grads = grads.detach().cpu()

    # ----------------------------------------------------------------------------------------------------
    # Path to save saliency maps
    save_path = os.path.join(save_dir, 'saliency')
    # Path to save mask (top 50% salient regions)
    mask_path = os.path.join(save_dir, 'mask_top50')
    # ----------------------------------------------------------------------------------------------------
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    for filename, grad in zip(filenames, grads):
        std = 3 * grad.std()
        saliency = (1 + grad / std) * 0.5
        saliency = np.clip(saliency, 0, 1)
        torchvision.utils.save_image(saliency, os.path.join(save_path, filename))

        # Convert saliency to grayscale (e.g., by averaging across channels)
        saliency_gray = saliency.mean(dim=0)  # [H, W]

        # ----------------------------------------------------------------------------------------------------
        # Set threshold: for example, top 80% as salient region

        threshold = saliency_gray.quantile(0.8)  # 80th percentile

        mask = (saliency_gray >= threshold).float()  # [H, W], values are 0 or 1

        # ----------------------------------------------------------------------------------------------------

        # Optional: Save the mask as an image
        torchvision.utils.save_image(mask.unsqueeze(0), os.path.join(mask_path, filename.split('.')[0] + '.png'))
