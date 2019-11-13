from torch import optim


def create_model(model_name, input_channels, num_classes, device, filters=64):

    possible_models = ['UNet2D', 'AttentionUnet2DSingleGating', 'AttentionUnet2DMultiGating']
    model_name = model_name.lower()
    if model_name == 'unet2d':
        from models.UNet2D import UNet
        net = UNet(num_channels=input_channels, num_classes=num_classes, filters=filters)
        net = net.to(device)
    elif model_name == 'attentionunet2dsinglegating':
        from models.Attention_UNet_Common_Gating_2D import Attention_UNet
        net = Attention_UNet(num_channels=input_channels, num_classes=num_classes, filters=filters)
        net = net.to(device)
    elif model_name == 'attentionunet2dmultigating':
        from models.Attention_UNet_Many_Gatings_2D import Attention_UNet
        net = Attention_UNet(num_channels=input_channels, num_classes=num_classes, filters=filters)
        net = net.to(device)
    else:
        error_str = 'Model Architecture is either unknown or yet to be implemented, try one of these options instead:'
        for m in possible_models:
            error_str += '\n{}'.format(m)
        raise NotImplementedError(error_str)

    return net


def create3Dmodel(model_name, input_channels, num_classes, device, filters=32):

    possible_models = ['UNet3D', 'AttentionUnet3DSingleGating', 'AttentionUnet3DMultiGating']
    model_name = model_name.lower()
    if model_name == 'unet3d':
        from models.UNet3D import UNet
        net = UNet(num_channels=input_channels, num_classes=num_classes, filters=filters)
        net = net.to(device)
    elif model_name == 'attentionunet3dsinglegating':
        from models.Attention_UNet_Common_Gating_3D import Attention_UNet
        net = Attention_UNet(num_channels=input_channels, num_classes=num_classes, filters=filters)
        net = net.to(device)
    elif model_name == 'attentionunet3dmultigating':
        from models.Attention_UNet_Many_Gatings_3D import Attention_UNet
        net = Attention_UNet(num_channels=input_channels, num_classes=num_classes, filters=filters)
        net = net.to(device)
    else:
        error_str = 'Model Architecture is either unknown or yet to be implemented, try one of these options instead:'
        for m in possible_models:
            error_str += '\n{}'.format(m)
        raise NotImplementedError(error_str)

    return net


def create_loss_criterion(loss_type):

    possible_losses = ['BCE', 'SoftDiceLoss2D']
    loss_type = loss_type.lower()
    if loss_type == 'bce':
        from torch.nn import BCELoss
        loss_criterion = BCELoss()
    elif loss_type == 'softdiceloss2d':
        from losses import SoftDiceLoss2D
        loss_criterion = SoftDiceLoss2D()
    else:
        error_str = 'Optimizer parameter is either unknown or yet to be implemented, try one of these options instead:'
        for m in possible_losses:
            error_str += '\n{}'.format(m)
        raise NotImplementedError(error_str)

    return loss_criterion


def create_optimizer(optimizer_name, net_params, eta, momentum=0.99, weight_decay= 0.0005):

    possible_models = ['SGD', 'Adam']
    name = optimizer_name.lower()
    if name == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net_params), lr=eta, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net_params), lr=eta, weight_decay=weight_decay)
    else:
        error_str = 'Optimizer parameter is either unknown or yet to be implemented, try one of these options instead:'
        for m in possible_models:
            error_str += '\n{}'.format(m)
        raise NotImplementedError(error_str)

    return optimizer


def update_optimizer_learning_rate(optimizer, new_eta):

    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] == new_eta
    optimizer.load_state_dict[state_dict]

    return optimizer


def create_dataset(data_format, train_path, test_path):

    possible_formats = ['Image', 'Numpy', 'Nifti']
    data_format = data_format.lower()
    if data_format == 'image':
        from custom_data_loader import DataSet
        dataset = DataSet(train_path, test_path)
    elif data_format == 'numpy':
        from custom_data_loader import NumpyDataset
        dataset = NumpyDataset(train_path, test_path)
    elif data_format == 'nifti':
        from custom_data_loader import NiftiDataset
        dataset = NiftiDataset(train_path, test_path)
    else:
        error_str = 'Data format is either unknown or yet to be implemented, try one of these options instead:'
        for m in possible_formats:
            error_str += '\n{}'.format(m)
        raise NotImplementedError(error_str)

    return dataset


def create_LUNA_datasets(subsets_folder):

    import os
    from custom_data_loader import NumpyDataset
    paths = os.listdir(subsets_folder)
    paths = sorted(x for x in paths if 'resampled' in x)

    datasets = []

    for path in paths:

        for elem in os.listdir(os.path.join(subsets_folder, path)):

            if 'Masks' in elem:

                test_path = os.path.join(subsets_folder, path, elem)

            else:

                train_path = os.path.join(subsets_folder, path, elem)

        dataset = NumpyDataset(train_path, test_path)
        datasets.append(dataset)

    return datasets