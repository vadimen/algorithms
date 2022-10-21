/*
 ============================================================================
 Name        :
 Author      : Vadim Placinta
 Version     :
 Copyright   : TalTech
 Description : machine learning for embedded systems homework nr 1
 a small nn network with backprop, but I made it a small generic library
 ============================================================================
 14 - 17 oct 2022: created, worked ~8 hours a day 
 21 oct 2022: tested everything, error is not in implementation, smth logical
 error function is wrongly implemented for the case when there is more than 1 output neuron
 */

#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"
#include "time.h"
#include "math.h"
#include <string.h>

#define RANDOM 0
#define PRECISE 1

struct NN_Descriptor
{
    int nr_layers; // inputs are not taken as a layer
    int input_size;
    int *every_layer_size; // all hidden and outputs are taken as a layer
    int output_size;

    // for single feedforward step
    double **neuron_values_arrs;
    double **weights_arrs;
    double **biases_arrs;

    // for single backpropagation step
    // this arrs will store derivs of every param regarding cost funct 
    // no accumulation happens here, on the next backprop they are reinitialized
    double **neuron_derivs_arrs;
    double **activ_func_of_z_derivs_arrs;
    double **biases_derivs_arrs;
    double **weights_derivs_arrs;

    //this are basically same arrs as above but they accumulate
    //all the derivs in order to get an average value later
    double **acc_neuron_derivs_arrs;
    //double **acc_activ_func_of_z_derivs_arrs;
    double **acc_biases_derivs_arrs;
    double **acc_weights_derivs_arrs;
};

struct Dataset_Descriptor{
    int test_set_rows;
    int train_set_rows;
    int dataset_input_cols;
    int dataset_output_cols;
    double **test_set_inputs;
    double **test_set_outputs;
    double **train_set_inputs;
    double **train_set_outputs;
};

int print_m_by_n_matrix(double **arr_in, int m, int n);
int alloc_m_by_n_matrix(double **arr_in, int m, int n, int type, double value);
double randfrom(double min, double max);
int show_nn_matrices(struct NN_Descriptor *nn);
double relu(double value);
double relu_der(double value);
double sigmoid(double value);
double sigmoid_der(double value);
double square_diff(double a, double b);
double square_diff_der(double a, double b, double output_size);
int feedforward_step(struct NN_Descriptor *nn, double *input);
int calc_all_derivs_step(struct NN_Descriptor *nn, double *inputs, double *ground_truth);
int add_derivs_to_accummulators(struct NN_Descriptor *nn);
int divide_derivs_in_accumms_by_batch(struct NN_Descriptor *nn, int batch_size);
int update_all_params(struct NN_Descriptor *nn, double learning_rate);
void split_string(char *line, double **dataset, int cols);
int read_dataset(char *file_name, double **dataset, int cols);
int alloc_all_nn_matrices(struct NN_Descriptor *nn);
int create_xor_dataset(struct Dataset_Descriptor *dts);
int train(struct NN_Descriptor *nn, struct Dataset_Descriptor *dts, int nr_epochs, int batch_size, double lr);
int perform_one_epoch(struct NN_Descriptor *nn, struct Dataset_Descriptor *dts, int batch_size, double lr);
int test_neural_net(struct NN_Descriptor *nn, struct Dataset_Descriptor *dts);

// //cancer set
// int test_set_rows = 114;
// int train_set_rows = 455;
// int dataset_cols = 31;
// double test_set[114][31];
// double train_set[455][31];

int main(void)
{
    //srand(17);
    struct NN_Descriptor nn;
    struct Dataset_Descriptor xor_dataset;

    //course network
    // nn.nr_layers = 3;//inputs are not taken as a layer
    // nn.input_size = 2;
    // nn.every_layer_size = (int*)malloc(nn.nr_layers * sizeof(int));
    // nn.every_layer_size[0] = 5;
    // nn.every_layer_size[1] = 4;
    // nn.every_layer_size[2] = 1;//outputs are taken as a layer
    // nn.output_size = nn.every_layer_size[2];

    // my test network
    nn.nr_layers = 5; // inputs are not taken as a layer
    nn.input_size = 2;
    nn.every_layer_size = (int *)malloc(nn.nr_layers * sizeof(int));
    nn.every_layer_size[0] = 5;
    nn.every_layer_size[1] = 5;
    nn.every_layer_size[2] = 5;
    nn.every_layer_size[3] = 5;
    nn.every_layer_size[4] = 1;
    nn.output_size = nn.every_layer_size[4];

    // cancer network
    // nn.nr_layers = 5; // inputs are not taken as a layer
    // nn.input_size = dataset_cols - 1; //30
    // nn.every_layer_size = (int *)malloc(nn.nr_layers * sizeof(int));
    // nn.every_layer_size[0] = 5;
    // nn.every_layer_size[1] = 10;
    // nn.every_layer_size[2] = 10;
    // nn.every_layer_size[3] = 5;
    // nn.every_layer_size[4] = 2;
    // nn.output_size = nn.every_layer_size[ nn.nr_layers-1 ];

    //===================================================================
    alloc_all_nn_matrices(&nn);

    create_xor_dataset(&xor_dataset);

    show_nn_matrices(&nn);

    train(&nn, &xor_dataset, 10000, 1, 0.00001);
    test_neural_net(&nn, &xor_dataset);

    //show_nn_matrices(&nn);
}

int create_xor_dataset(struct Dataset_Descriptor *dts){

    dts->test_set_rows = 2;
    dts->train_set_rows = 4;
    dts->dataset_input_cols = 2;
    dts->dataset_output_cols = 1;

    //train set
    dts->train_set_inputs = (double**)malloc(dts->train_set_rows * sizeof(double*) );
    dts->train_set_outputs = (double**)malloc(dts->train_set_rows * sizeof(double*) );

    dts->train_set_inputs[0] = (double*)malloc(dts->dataset_input_cols * sizeof(double));
    dts->train_set_inputs[1] = (double*)malloc(dts->dataset_input_cols * sizeof(double));
    dts->train_set_inputs[2] = (double*)malloc(dts->dataset_input_cols * sizeof(double));
    dts->train_set_inputs[3] = (double*)malloc(dts->dataset_input_cols * sizeof(double));
    
    dts->train_set_outputs[0] = (double*)malloc(dts->dataset_output_cols * sizeof(double));
    dts->train_set_outputs[1] = (double*)malloc(dts->dataset_output_cols * sizeof(double));
    dts->train_set_outputs[2] = (double*)malloc(dts->dataset_output_cols * sizeof(double));
    dts->train_set_outputs[3] = (double*)malloc(dts->dataset_output_cols * sizeof(double));

    dts->train_set_inputs[2][0] = 0; dts->train_set_inputs[2][1] = 0; dts->train_set_outputs[2][0] = 0;
    dts->train_set_inputs[1][0] = 0; dts->train_set_inputs[1][1] = 1; dts->train_set_outputs[1][0] = 1;
    dts->train_set_inputs[0][0] = 1; dts->train_set_inputs[0][1] = 0; dts->train_set_outputs[0][0] = 1;
    dts->train_set_inputs[3][0] = 1; dts->train_set_inputs[3][1] = 1; dts->train_set_outputs[3][0] = 0;

    //test set
    dts->test_set_inputs = (double**)malloc(dts->test_set_rows * sizeof(double*) );
    dts->test_set_outputs = (double**)malloc(dts->test_set_rows * sizeof(double*) );

    dts->test_set_inputs[0] = (double*)malloc(dts->dataset_input_cols * sizeof(double));
    dts->test_set_inputs[1] = (double*)malloc(dts->dataset_input_cols * sizeof(double));
    
    dts->test_set_outputs[0] = (double*)malloc(dts->dataset_output_cols * sizeof(double));
    dts->test_set_outputs[1] = (double*)malloc(dts->dataset_output_cols * sizeof(double));

    dts->test_set_inputs[0][0] = 0; dts->test_set_inputs[0][1] = 0; dts->test_set_outputs[0][0] = 0;
    dts->test_set_inputs[1][0] = 0; dts->test_set_inputs[1][1] = 1; dts->test_set_outputs[1][0] = 1;

}

int test_neural_net(struct NN_Descriptor *nn, struct Dataset_Descriptor *dts){

    for(int step=0; step < dts->test_set_rows; step++){
        double *inputs = dts->test_set_inputs[step];
        double *ground_truth = dts->test_set_outputs[step];

        printf("the inputs\n");
        print_m_by_n_matrix(&inputs, 1, dts->dataset_input_cols);

        feedforward_step(nn, inputs);

        printf("\tpredicted\ttrue\n");
        for(int out_neuron=0; out_neuron<nn->every_layer_size[nn->nr_layers-1];out_neuron++){
            printf("\t%f\t%f\n", nn->neuron_values_arrs[nn->nr_layers-1][out_neuron], ground_truth[out_neuron]);
        }
    }
    printf("\n");
}

int train(struct NN_Descriptor *nn, struct Dataset_Descriptor *dts, int nr_epochs, int batch_size, double lr){
    for(int epoch=0; epoch<nr_epochs; epoch++){
        printf("running epoch %d...\n", epoch);
        perform_one_epoch(nn, dts, batch_size, lr);
    }
}

int perform_one_epoch(struct NN_Descriptor *nn, struct Dataset_Descriptor *dts, int batch_size, double lr){
    
    //for a progress bar
    int one_percent_of_rows = (int)(dts->train_set_rows / 100.0);
    if(one_percent_of_rows < 1) one_percent_of_rows = 1;


    for(int step=0; step < dts->train_set_rows; step++){
        double *inputs = dts->train_set_inputs[step];
        double *ground_truth = dts->train_set_outputs[step];

        feedforward_step(nn, inputs);
        calc_all_derivs_step(nn, inputs, ground_truth);
        add_derivs_to_accummulators(nn);

        if(batch_size == 1 || step % batch_size == 0 || step == dts->train_set_rows-1){
            divide_derivs_in_accumms_by_batch(nn, batch_size);
            update_all_params(nn, lr);
        }

        if(dts->train_set_rows < 100 || step % one_percent_of_rows == 0) printf("|");
    }
    printf("\n");
}

int alloc_all_nn_matrices(struct NN_Descriptor *nn){
    // arrs needed for feedforward
    nn->neuron_values_arrs = (double **)malloc(nn->nr_layers * sizeof(double *));
    nn->weights_arrs = (double **)malloc(nn->nr_layers * sizeof(double *));
    nn->biases_arrs = (double **)malloc(nn->nr_layers * sizeof(double *));

    // arrs needed for single backprop
    nn->neuron_derivs_arrs = (double **)malloc(nn->nr_layers * sizeof(double *));
    nn->activ_func_of_z_derivs_arrs = (double **)malloc(nn->nr_layers * sizeof(double *));
    nn->biases_derivs_arrs = (double **)malloc(nn->nr_layers * sizeof(double *));
    nn->weights_derivs_arrs = (double **)malloc(nn->nr_layers * sizeof(double *));

    // arrs needed for backprop accumulation
    nn->acc_neuron_derivs_arrs = (double **)malloc(nn->nr_layers * sizeof(double *));
    nn->acc_biases_derivs_arrs = (double **)malloc(nn->nr_layers * sizeof(double *));
    nn->acc_weights_derivs_arrs = (double **)malloc(nn->nr_layers * sizeof(double *));

    for (int i = 0; i < nn->nr_layers; i++)
    {
        printf("allocating arr for feedforward:\n");
        printf("allocating neuron values:\n");
        alloc_m_by_n_matrix(nn->neuron_values_arrs + i, nn->every_layer_size[i], 1, RANDOM, 0);

        printf("allocating biases:\n");
        alloc_m_by_n_matrix(nn->biases_arrs + i, nn->every_layer_size[i], 1, RANDOM, 0);

        printf("allocating weights matrices:\n");
        int prev_layer_size;
        if (i - 1 < 0)
            prev_layer_size = nn->input_size;
        else
            prev_layer_size = nn->every_layer_size[i - 1];

        alloc_m_by_n_matrix(nn->weights_arrs + i, nn->every_layer_size[i], prev_layer_size, RANDOM, 0);

        printf("allocating arrs for backprop...\n");
        printf("allocating nn.neuron_derivs_arrs:\n");
        alloc_m_by_n_matrix(nn->neuron_derivs_arrs + i, nn->every_layer_size[i], 1, PRECISE, 0);
        printf("allocating nn.activ_func_of_z_derivs_arrs:\n");
        alloc_m_by_n_matrix(nn->activ_func_of_z_derivs_arrs + i, nn->every_layer_size[i], 1, PRECISE, 0);
        printf("allocating nn.biases_derivs_arrs:\n");
        alloc_m_by_n_matrix(nn->biases_derivs_arrs + i, nn->every_layer_size[i], 1, PRECISE, 0);
        printf("allocating nn.weights_derivs_arrs:\n");
        alloc_m_by_n_matrix(nn->weights_derivs_arrs + i, nn->every_layer_size[i], prev_layer_size, PRECISE, 0);


        printf("allocating arrs for backprop accumulation...\n");
        alloc_m_by_n_matrix(nn->acc_neuron_derivs_arrs + i, nn->every_layer_size[i], 1, PRECISE, 0);
        alloc_m_by_n_matrix(nn->acc_biases_derivs_arrs + i, nn->every_layer_size[i], 1, PRECISE, 0);
        alloc_m_by_n_matrix(nn->acc_weights_derivs_arrs + i, nn->every_layer_size[i], prev_layer_size, PRECISE, 0);
    }
}

int update_all_params(struct NN_Descriptor *nn, double learning_rate){
    //going backwards as in backprop, no diff if going other dir thought
    for (int layer = nn->nr_layers - 1; layer >= 0; layer--){
        for (int neuron = 0; neuron < nn->every_layer_size[layer]; neuron++){
            //nn->neuron_values_arrs[layer][neuron] -= learning_rate * nn->acc_neuron_derivs_arrs[layer][neuron];
            nn->biases_arrs[layer][neuron] -= learning_rate * nn->acc_biases_derivs_arrs[layer][neuron];

            int prev_layer_size;
            if (layer == 0){
                prev_layer_size = nn->input_size;
            }else{
                prev_layer_size = nn->every_layer_size[layer - 1];
            }

            for (int wire_into = 0; wire_into < prev_layer_size; wire_into++){
                nn->weights_arrs[layer][neuron * prev_layer_size + wire_into] -= 
                                    learning_rate * nn->acc_weights_derivs_arrs[layer][neuron * prev_layer_size + wire_into];
            }
        }
    }
    return 0;
}

int divide_derivs_in_accumms_by_batch(struct NN_Descriptor *nn, int batch_size){
    //going backwards as in backprop, no diff if going other dir thought
    for (int layer = nn->nr_layers - 1; layer >= 0; layer--){
        for (int neuron = 0; neuron < nn->every_layer_size[layer]; neuron++){

            nn->acc_neuron_derivs_arrs[layer][neuron] /= (double)batch_size;
            nn->acc_biases_derivs_arrs[layer][neuron] /= (double)batch_size;

            int prev_layer_size;
            if (layer == 0)
                prev_layer_size = nn->input_size;
            else
                prev_layer_size = nn->every_layer_size[layer - 1];

            for (int wire_into = 0; wire_into < prev_layer_size; wire_into++)
                nn->acc_weights_derivs_arrs[layer][neuron * prev_layer_size + wire_into] /= (double)batch_size;
        }
    }
    return 0;
}

int add_derivs_to_accummulators(struct NN_Descriptor *nn){
    //going backwards as in backprop, no diff if going other dir thought
    for (int layer = nn->nr_layers - 1; layer >= 0; layer--){
        for (int neuron = 0; neuron < nn->every_layer_size[layer]; neuron++){

            nn->acc_neuron_derivs_arrs[layer][neuron] += nn->neuron_derivs_arrs[layer][neuron];
            nn->acc_biases_derivs_arrs[layer][neuron] += nn->biases_derivs_arrs[layer][neuron];

            int prev_layer_size;
            if (layer == 0){
                prev_layer_size = nn->input_size;
            }else{
                prev_layer_size = nn->every_layer_size[layer - 1];
            }

            for (int wire_into = 0; wire_into < prev_layer_size; wire_into++){
                nn->acc_weights_derivs_arrs[layer][neuron * prev_layer_size + wire_into] += 
                                    nn->weights_derivs_arrs[layer][neuron * prev_layer_size + wire_into];
            }
        }
    }
    return 0;
}

int calc_all_derivs_step(struct NN_Descriptor *nn, double *inputs, double *ground_truth){
    for (int layer = nn->nr_layers - 1; layer >= 0; layer--){
        for (int neuron = 0; neuron < nn->every_layer_size[layer]; neuron++){
            // if we are on the output layer
            // we calc its deriv right here
            if(layer == nn->nr_layers - 1){
                nn->neuron_derivs_arrs[layer][neuron] = 
                        square_diff_der(nn->neuron_values_arrs[layer][neuron], ground_truth[neuron], nn->output_size);
            }else{
                //because we already accumulated derivs of this neuron in regard to every neuron from next layer
                //we now divide it by nr of neurons from next layer from which we accumulated derivs
                nn->neuron_derivs_arrs[layer][neuron] /= nn->every_layer_size[layer+1];
            }

            //we start with bias deriv, because we'll calc w and input neuron derivs using it
            nn->biases_derivs_arrs[layer][neuron] = nn->neuron_derivs_arrs[layer][neuron] *
                                                    nn->activ_func_of_z_derivs_arrs[layer][neuron];

            //if it's layer 0(first one after input layer), we derive weights in respect with input
            //and use the size of the input layer
            int prev_layer_size;
            double *prev_neuron_values;
            if (layer == 0){
                prev_layer_size = nn->input_size;
                prev_neuron_values = inputs;
            }else{
                prev_layer_size = nn->every_layer_size[layer - 1];
                prev_neuron_values = nn->neuron_values_arrs[layer - 1];
            }

            // calculating derivs of all weights and neurons connected into this neuron
            for (int wire_into = 0; wire_into < prev_layer_size; wire_into++){
                //if prev neuron layer is the input layers
                //we do not need to calc derivs of inputs, only derivs of weights connected to them
                //here we accummulate all derivs in the corresponding layer, in next iteration we will divide
                if(layer > 0){
                    nn->neuron_derivs_arrs[layer - 1][wire_into] += ( nn->biases_derivs_arrs[layer][neuron] *
                                                                nn->weights_arrs[layer][neuron * prev_layer_size + wire_into] );
                }

                nn->weights_derivs_arrs[layer][neuron * prev_layer_size + wire_into] = nn->biases_derivs_arrs[layer][neuron] *
                                                            prev_neuron_values[wire_into];
            }
        }
    }
    return 0;
}

/*
 ============================================================================
 Description:
 Input:
 Output:
 ============================================================================
 */
int feedforward_step(struct NN_Descriptor *nn, double *input)
{
    // do this for how many neuron layers we have, except input
    for (int layer = 0; layer < nn->nr_layers; layer++)
    {

        // multiplying every weight of each neuron from this layers
        // with every neuron value from previous neuron layer or inputs if that's the case
        for (int neuron = 0; neuron < nn->every_layer_size[layer]; neuron++)
        {
            double this_neuron_z = 0;
            // every neuron has as many weights as there are neurons in prev layer
            int prev_layer_size;
            double *prev_neuron_values;
            if (layer - 1 < 0)
            {
                prev_layer_size = nn->input_size;
                prev_neuron_values = input;
            }
            else
            {
                prev_layer_size = nn->every_layer_size[layer - 1];
                prev_neuron_values = nn->neuron_values_arrs[layer - 1];
            }

            for (int weight_nr = 0; weight_nr < prev_layer_size; weight_nr++)
            {
                this_neuron_z +=
                    nn->weights_arrs[layer][neuron * prev_layer_size + weight_nr] *
                        prev_neuron_values[weight_nr];
            }

            this_neuron_z += nn->biases_arrs[layer][neuron];

            //sigmoid for output layer
            //relu for hidden layers
            if(layer == nn->nr_layers-1){
                nn->neuron_values_arrs[layer][neuron] = sigmoid(this_neuron_z);
                nn->activ_func_of_z_derivs_arrs[layer][neuron] = sigmoid_der(this_neuron_z);
            }else{
                nn->neuron_values_arrs[layer][neuron] = relu(this_neuron_z);
                nn->activ_func_of_z_derivs_arrs[layer][neuron] = relu_der(this_neuron_z);
            }
        }
    }
}

double square_diff(double a, double b)
{
    return (a - b) * (a - b);
}

double square_diff_der(double a, double b, double output_size)
{
    return 2.0 / (double)output_size * (a - b);
}

double relu(double value)
{
    if (value < 0)
        return 0;
    else
        return value;
}

double relu_der(double value){
    if(value < 0)
        return 0;
    else
        return 1;
}

double sigmoid(double value)
{
    double result;
    result = 1 / (1 + exp(-value));
    return result;
}

double sigmoid_der(double value)
{
    double sigm_val = sigmoid(value);
    return sigm_val * (1 - sigm_val);
}

int show_nn_matrices(struct NN_Descriptor *nn)
{
    printf("======================START SHOWING THE MATRICES========================\n");
    for (int i = 0; i < nn->nr_layers; i++)
    {
        printf("============= layer nr %d ==============\n", i);
        printf("neuron values:\n");
        print_m_by_n_matrix(nn->neuron_values_arrs + i, nn->every_layer_size[i], 1);

        printf("biases:\n");
        print_m_by_n_matrix(nn->biases_arrs + i, nn->every_layer_size[i], 1);

        printf("weights matrices:\n");
        int prev_layer_size;
        if (i - 1 < 0)
            prev_layer_size = nn->input_size;
        else
            prev_layer_size = nn->every_layer_size[i - 1];

        print_m_by_n_matrix(nn->weights_arrs + i, nn->every_layer_size[i], prev_layer_size);

        printf("nn.neuron_derivs_arrs:\n");
        print_m_by_n_matrix(nn->neuron_derivs_arrs + i, nn->every_layer_size[i], 1);
        printf("nn.activ_func_of_z_derivs_arrs:\n");
        print_m_by_n_matrix(nn->activ_func_of_z_derivs_arrs + i, nn->every_layer_size[i], 1);
        printf("nn.biases_derivs_arrs:\n");
        print_m_by_n_matrix(nn->biases_derivs_arrs + i, nn->every_layer_size[i], 1);
        printf("nn.weights_derivs_arrs:\n");
        print_m_by_n_matrix(nn->weights_derivs_arrs + i, nn->every_layer_size[i], prev_layer_size);
    }
    printf("======================END SHOWING THE MATRICES========================\n");
}

/*
 ============================================================================
 Description:
 Input:
 Output:
 ============================================================================
 */
int alloc_m_by_n_matrix(double **arr_in, int m, int n, int type, double value)
{
    double *arr = (double *)malloc(m * n * sizeof(double));

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            *(arr + i * n + j) = type == PRECISE ? value : randfrom(0, 1.0);

    *arr_in = arr;

    printf("we allocated an %dx%d matrix:\n", m, n);
    print_m_by_n_matrix(arr_in, m, n);

    return 0;
}

/*
 ============================================================================
 Description:
 Input:
 Output:
 ============================================================================
 */
int print_m_by_n_matrix(double **arr_in, int m, int n)
{
    double *arr = *arr_in;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%f ", *(arr + i * n + j));
        }
        printf("\n\n");
    }
    printf("\n");

    return 0;
}

// from stackoverflow
double randfrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

// // partly from stackoverflow
// void split_string(char *line, double **rowp, int cols) {
//     double *data = (double*)malloc(cols * sizeof(double));

//     const char delimiter[] = "\t";
//     char *tmp;
//     int col = 0;

//     tmp = strtok(line, delimiter);
//     if (tmp == NULL)
//     return;

//     data[col] = atof(tmp); 
//     col++;

//     for (;;) {
//         tmp = strtok(NULL, delimiter);
//         if (tmp == NULL)
//             break;
//         data[col] = atof(tmp); 
//         col++;
//     }

//     *rowp = data;
// }

// // partly from stackoverflow
// int read_dataset(char *file_name, double **dataset, int cols){ //22 cols
//     FILE * fp;
//     char * line = NULL;
//     size_t len = 0;
//     ssize_t read;

//     fp = fopen(file_name, "r");
//     if (fp == NULL){
//         printf("failure\n");
//         return 0;
//     }

//     int row = 0;

//     while ((read = getline(&line, &len, fp)) != -1) {
//         split_string(line, dataset+row, cols);
//         row++;
//     }

//     fclose(fp);
//     if (line)
//         free(line);
// }