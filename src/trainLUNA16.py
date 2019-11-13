import torch
import numpy as np
import copy
from evaluation_metrics import dice_similarity_score
from utils import json_file_to_pyobj
import os
from create_training_objects import create3Dmodel, create_optimizer, create_loss_criterion, create_LUNA_datasets
from k_folds_creator import create_k_LUNA_folds
from evaluation_metrics import specificity_LUNA, recall_sensitivity_LUNA


def train(args):
    
    json_options = json_file_to_pyobj(args.config)
    training_configurations = json_options.training

    eta = training_configurations.eta
    epochs = training_configurations.epochs
    batch_size = training_configurations.batch_size
    patience = training_configurations.patience
    k, number_of_k_runs = training_configurations.k_folds, training_configurations.k_fold_iter

    loss_criterion = create_loss_criterion(training_configurations.loss_type)
    output_directory = training_configurations.output_file_directory

    if output_directory == "":
        print('##########################################################')
        print('Eta:{}, Epochs: {}, Batch Size: {}, Folds: {}, Runs: {}'.format(eta, epochs, batch_size, k, number_of_k_runs))
        print('##########################################################')
        print()
        use_output = False
    else:
        output_file_name = '{}_{}folds_{}runs_{}epochs_{}_eta_{}batch_{}_loss_{}optimizer.txt'.format(training_configurations.model, k, number_of_k_runs, epochs, str(eta).replace('.', '-'), batch_size, training_configurations.loss_type, training_configurations.optimizer)
        output_file = os.path.join(output_directory, output_file_name)
        with open(output_file, 'w') as outfile:
            outfile.write('##########################################################\n')
            outfile.write('Eta:{}, Epochs: {}, Batch Size: {}, Folds: {}, Runs: {}\n'.format(eta, epochs, batch_size, k, number_of_k_runs))
            outfile.write('##########################################################\n')
            outfile.write('\n')
            outfile.close()
            use_output = True

    device = torch.device('cuda:{}'.format(training_configurations.gpu))
    net = create3Dmodel(training_configurations.model, training_configurations.input_channels, training_configurations.num_classes, device)

    path = training_configurations.path
    datasets = create_LUNA_datasets(path)

    save_init_model = copy.deepcopy(net)
    best_model = copy.deepcopy(net)

    loaders = create_k_LUNA_folds(datasets, k, batch_size)

    all_acc, test_set_dice_similarities = [], []
    all_sensitivities, test_set_sensitivities,  = [], []
    all_specificities, test_set_specificities,  = [], []

    for k_fold_run in range(number_of_k_runs):

        torch.cuda.empty_cache()
        for fold_index, (train_loader, test_loader) in enumerate(loaders):

            if k_fold_run == 0:
                if fold_index > 0:
                    del k_net_model
                    del best_model
                    torch.cuda.empty_cache()
            else:
                del k_net_model
                del best_model
                torch.cuda.empty_cache()

                if fold_index > 4:
                    epochs = 50
                else:
                    epochs = 0

            k_net_model = copy.deepcopy(save_init_model)

            k_fold_optimizer = create_optimizer(training_configurations.optimizer, k_net_model.parameters(), eta)

            epoch_break = 0
            best_score = 0

            patience_acc = 0
            epoch_accuracies = [0]

            for epoch in range(epochs):

                update_step_accuracy = []

                k_net_model.train()
                for _, (images, gts) in enumerate(train_loader):
                    next(iter(train_loader))[0]

                    images = images.float().to(device)
                    gts = gts.float().to(device)

                    k_fold_optimizer.zero_grad()
                    preds = k_net_model(images)

                    loss = loss_criterion(preds, gts)
                    dice_similarity = dice_similarity_score(preds, gts)
                    update_step_accuracy.append(dice_similarity)

                    loss.backward()
                    k_fold_optimizer.step()

                epoch_accuracy = np.average(update_step_accuracy)
                if not use_output:
                    print('Epoch {} accuracy: {}%'.format(epoch+1, round(100*epoch_accuracy, 2)))
                else:
                    with open(output_file, 'a') as outfile:
                        outfile.write('Epoch {} accuracy: {}%\n'.format(epoch+1, round(100*epoch_accuracy, 2)))

                if epoch_accuracy > best_score:

                    epoch_break = epoch
                    patience_acc = 0
                    best_score = epoch_accuracy
                    best_model = copy.deepcopy(k_net_model)
                    # torch.save(best_model.state_dict(), './../checkpoints/best_dice_chest_xray_exp{}.pt'.format(experiment_index))

                elif epoch_accuracies[-1] > epoch_accuracy:

                    patience_acc += 1
                    if patience_acc == patience:
                        break

                else:
                    patience_acc = 0

                epoch_accuracies.append(epoch_accuracy)

            test_set_dices = []
            test_set_sensitivities= []
            test_set_specificities = []

            k_net_model.eval()
            for _, (test_images, test_gts) in enumerate(test_loader):

                with torch.no_grad():
                    test_images = test_images.float().to(device)
                    test_gts = test_gts.float().to(device)

                    test_preds = best_model(test_images)

                    test_sample_dice_similarity = dice_similarity_score(test_preds, test_gts)
                    sensitivity = recall_sensitivity_LUNA(test_preds, test_gts)
                    specificity_measure = specificity_LUNA(test_preds, test_gts)

                    test_set_dices.append(test_sample_dice_similarity)
                    test_set_sensitivities.append(sensitivity)
                    test_set_specificities.append(specificity_measure)

            fold_acc = np.average(test_set_dices)
            fold_sensitivity= np.average(test_set_sensitivities)
            fold_specificity= np.average(test_set_specificities)

            all_acc.extend(test_set_dices)
            all_sensitivities.extend(test_set_sensitivities)
            all_specificities.extend(test_set_specificities)

            if epoch_break == 0:
                epoch_break = epochs-1

            if not use_output:
                print('##########################################################')
                print('Completed fold no.{} in epoch {} with test set Dice Similarity: {}'.format(fold_index+1, epoch_break+1, round(100*fold_acc, 2)))
                print('Completed fold no.{} in epoch {} with test set Sensitivity: {}'.format(fold_index+1, epoch_break+1, round(fold_sensitivity, 5)))
                print('Completed fold no.{} in epoch {} with test set specificity_LUNA: {}'.format(fold_index+1, epoch_break+1, round(fold_specificity, 5)))
                print('##########################################################')
                print()
            else:
                with open(output_file, 'a') as outfile:
                    outfile.write('##########################################################\n')
                    outfile.write('Completed fold no.{} in epoch {} with test set dice similarity: {}\n'.format(fold_index+1, epoch_break+1, round(100*fold_acc, 2)))
                    outfile.write('Completed fold no.{} in epoch {} with test set Sensitivity: {}\n'.format(fold_index + 1, epoch_break + 1, round(fold_sensitivity, 5)))
                    outfile.write('Completed fold no.{} in epoch {} with test set specificity_LUNA: {}\n'.format(fold_index + 1, epoch_break + 1, round(fold_specificity, 5)))
                    outfile.write('##########################################################\n')
                    outfile.write('\n')
                    outfile.close()

        test_set_dice_similarity = 100*np.average(all_acc)
        test_set_sensitivity = np.average(all_sensitivities)
        test_set_specificity = np.average(all_specificities)

        test_set_dice_similarities.append(test_set_dice_similarity)
        test_set_sensitivities.append(test_set_sensitivity)
        test_set_specificities.append(test_set_specificity)

        if not use_output:
            print('##########################################################')
            print('MEAN TEST SET DICE SCORE OF {} FOLDS IN RUN {}: {}'.format(k, k_fold_run+1, round(test_set_dice_similarity, 2)))
            print('MEAN TEST SET SENSITIVITY OF {} FOLDS IN RUN {}: {}'.format(k, k_fold_run+1, round(test_set_sensitivity, 5)))
            print('MEAN TEST SET specificity_LUNA OF {} FOLDS IN RUN {}: {}'.format(k, k_fold_run+1, round(test_set_specificity, 5)))
            print('##########################################################')
            print()
        else:
            with open(output_file, 'a') as outfile:
                outfile.write('##########################################################\n')
                outfile.write('MEAN TEST DICE SCORE OF {} FOLDS IN RUN {}: {}\n'.format(k, k_fold_run + 1, round(test_set_dice_similarity, 2)))
                outfile.write('MEAN TEST SET SENSITIVITY OF {} FOLDS IN RUN {}: {}\n'.format(k, k_fold_run + 1, round(test_set_sensitivity, 5)))
                outfile.write('MEAN TEST SET specificity_LUNA OF {} FOLDS IN RUN {}: {}\n'.format(k, k_fold_run + 1, round(test_set_specificity, 5)))
                outfile.write('##########################################################\n')
                outfile.write('\n')
                outfile.close()

    average_test_set_dice_similarity = np.average(test_set_dice_similarities)
    average_test_set_sensitivity = np.average(test_set_sensitivities)
    average_test_set_specificity = np.average(test_set_specificities)

    if not use_output:
        print('##########################################################')
        print('AVERAGE {}-FOLD TEST SET DICE SIMILARITY IN {} RUNS : {}'.format(k, number_of_k_runs, round(average_test_set_dice_similarity, 2)))
        print('AVERAGE {}-FOLD TEST SET SENSITIVITY IN {} RUNS : {}'.format(k, number_of_k_runs, round(average_test_set_sensitivity, 5)))
        print('AVERAGE {}-FOLD TEST SET specificity_LUNA IN {} RUNS : {}'.format(k, number_of_k_runs, round(average_test_set_specificity, 5)))
        print('##########################################################')
        print()
    else:
        with open(output_file, 'a') as outfile:
            outfile.write('##########################################################\n')
            outfile.write('AVERAGE {}-FOLD TEST SET DICE SIMILARITY IN {} RUNS : {}\n'.format(k, number_of_k_runs, round(average_test_set_dice_similarity, 2)))
            outfile.write('AVERAGE {}-FOLD TEST SET SENSITIVITY IN {} RUNS : {}\n'.format(k, number_of_k_runs, round(average_test_set_sensitivity, 5)))
            outfile.write('AVERAGE {}-FOLD TEST SET specificity_LUNA IN {} RUNS : {}\n'.format(k, number_of_k_runs, round(average_test_set_specificity, 5)))
            outfile.write('##########################################################\n')
            outfile.write('\n')
            outfile.close()


if __name__ == '__main__':

    import argparse

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    parser = argparse.ArgumentParser(description='Chest X-Ray Lung Segmentation')

    parser.add_argument('-config', '--config', help='Training Configurations', required=True)

    args = parser.parse_args()

    train(args)
