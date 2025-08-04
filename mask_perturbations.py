def mask_perturbations(perturbations, args, filenames):
    ########################################################################
    # NT-Models/AT-Models
    mask_folder = f'./saliency_map/ARES_ResNet50_AT/{args.attack}/clean/mask'
    ########################################################################
    masks = []
    for _, filename in zip(range(perturbations.shape[0]), filenames):
        mask_path = os.path.join(mask_folder, filename.split('.')[0] + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = torch.tensor(mask, dtype=torch.float32)
        masks.append(mask)
    masks = torch.stack(masks)

    # Convert the mask to the same shape as the image
    # Expand the mask to [B, 1, H, W], then repeat C times to match the number of channels
    masks_expanded = masks.unsqueeze(1)  # Expand to [B, 1, H, W]
    masks_expanded = masks_expanded.repeat(1, perturbations.shape[1], 1, 1).to('cuda:0')  # Repeat C times to get [B, C, H, W]

    # Use the mask to remove corresponding parts from the image
    # Assume mask value 0 means the region should be removed
    ##############################################################################

    # Remove mask regions
    images_with_masked_part = perturbations * (1 - masks_expanded)

    ##############################################################################

    ############################################################################################################################################################
    # Save images after applying mask
    output_folder = f'./saliency_map/{args.model}/{args.attack}/clean/masked_front_perturbations_at_top20'  # masked_back_perturbations_at/masked_back_perturbations_nt
    os.makedirs(output_folder, exist_ok=True)

    # Save original perturbations for comparison before and after masking
    perturbations_output_path = f'./saliency_map/{args.model}/{args.attack}/clean/perturbations'  # perturbations_at/perturbations_nt
    os.makedirs(perturbations_output_path, exist_ok=True)
    ############################################################################################################################################################

    perturbs = (perturbations.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255).astype(np.uint8)
    masked_images = (images_with_masked_part.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255).astype(np.uint8)

    for i, filename in zip(range(perturbations.shape[0]), filenames):
        # Save original perturbation images
        filename = filename.split('.')[0] + '.png'
        Image.fromarray(perturbs[i]).save(os.path.join(perturbations_output_path, filename))
        # Save masked perturbation images
        Image.fromarray(masked_images[i]).save(os.path.join(output_folder, filename))

    return images_with_masked_part
