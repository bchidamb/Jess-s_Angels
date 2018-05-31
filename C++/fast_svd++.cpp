#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include "csv.h"

/*
 * DONT FORGET: compile with "-Ofast -march=native"
 */

#define n_users 458293
#define n_movies 17770
#define n_bins 30
#define beta 0.4

using namespace std;

struct Example {
    int m;
    int r;
};

struct UserData {
    vector<Example> entries;
    vector<int> R_u;
};

struct Dataset {
    vector<UserData> user_data;
    double mean;
    int n_examples;
};

// Based on https://stackoverflow.com/questions/29375797
template<typename T>
T** Dynamic2DArray(int w, int h) {
    T** arr = new T*[w];
    arr[0] = new T[w * h];
    for (int i = 1; i < w; i++) {
        arr[i] = arr[i-1] + h;
    }
    return arr;
}

class SVDpp {
    public:
    double mean; // average of all ratings (constant)
    double *b_u; // user biases
    double *b_m; // movie biases
    double **p; // user embeddings
    double **q; // movie embeddings
    double **y; // implicit latent factor
    
    double *sumY;  //
    double *gradY; // used to batch gradient updates of y for a given user
    
    int lf;

    SVDpp(int latent_factors) {
        lf = latent_factors;
        b_u = new double[n_users];
        b_m = new double[n_movies];
        p = Dynamic2DArray<double>(n_users, lf);
        q = Dynamic2DArray<double>(n_movies, lf);
        y = Dynamic2DArray<double>(n_movies, lf);
        
        sumY = new double[lf];
        gradY = new double[lf];

        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 0.2 / sqrt(lf));

        for (int i = 0; i < n_users; i++) {
            b_u[i] = 0.0;
            for (int j = 0; j < lf; j++) {
                p[i][j] = distribution(generator);
            }
        }
        for (int i = 0; i < n_movies; i++) {
            b_m[i] = 0.0;
            for (int j = 0; j < lf; j++) {
                q[i][j] = distribution(generator);
                y[i][j] = 0.0;
            }
        }
    }

    double pred_one(int &u, int &m, double &ru_negSqrt) {
        
        double * p_u = p[u];
        double * q_m = q[m];
        double dot = 0.0;
        for (int i = 0; i < lf; i++){ // which latent factor
            dot += (p_u[i] + ru_negSqrt * sumY[i]) * q_m[i];
        }
        return (dot + b_u[u] + b_m[m] + mean);
    }
    
    // u user, m movie
    double pred_one_test(int u, int m, vector<int> &r_u) {
        
        for(int i = 0; i < lf; i++) {
            sumY[i] = 0.0;
        }
        for (int r_u_j: r_u) {
            double *y_r_u_j = y[r_u_j];
            for (int i = 0; i < lf; i++) {
                sumY[i] += y_r_u_j[i];
            }
        }
        
        double *p_u = p[u];
        double *q_m = q[m];
        double dot = 0.0;
        double ru_negSqrt = (r_u.size() > 0) ? (1.0 / sqrt(r_u.size())) : 0.0;
        
        for (int i = 0; i < lf; i++){ // which latent factor
            dot += (p_u[i] + ru_negSqrt * sumY[i]) * q_m[i];
        }
        
        return (dot + b_u[u] + b_m[m] + mean);
    }

    void train(Dataset &data, Dataset &val_data, int epochs) {
        
        mean = data.mean;
        
        vector<int> idx;
        for (int i = 0; i < n_users; i++) {
            idx.push_back(i);
        }
        
        double decay = 1.0;
        for (int ep = 0; ep < epochs; ep++) {
            
            // cout << "Train RMSE: " << error(data, data) << endl;
            cout << "Val RMSE: " << error(val_data, data) << endl;
            cout << "Epoch " << ep << endl;
            random_shuffle(idx.begin(), idx.end());
            
            for (int u: idx) {
                vector<int> &R_u = data.user_data[u].R_u;
                double ru_negSqrt = (R_u.size() > 0) ? (1.0 / sqrt(R_u.size())) : 0.0;
                
                for(int j = 0; j < lf; j++) {
                    sumY[j] = 0.0;
                    gradY[j] = 0.0;
                }
                for (int R_u_k: R_u) {
                    for (int j = 0; j < lf; j++) {
                        sumY[j] += y[R_u_k][j];
                    }
                }
                
                random_shuffle(data.user_data[u].entries.begin(), data.user_data[u].entries.end());
                
                for (Example ex: data.user_data[u].entries) {
                    int m = ex.m;
                    
                    double dr = pred_one(u, m, ru_negSqrt) - ex.r;
                    
                    b_u[u] -= (0.007) * decay * (dr + (0.005) * b_u[u]);
                    b_m[m] -= (0.007) * decay * (dr + (0.005) * b_m[m]);
                    
                    double * q_m = q[m];
                    double * p_u = p[u];
                    
                    for (int j = 0; j < lf; j++) {
                        double p_old = p_u[j];
                        double q_old = q_m[j];
                        gradY[j] += dr * q_m[j] * ru_negSqrt;
                        p_u[j] -= (0.007) * decay * (dr * q_old + (0.015) * p_u[j]);
                        q_m[j] -= (0.007) * decay * (dr * (p_old + ru_negSqrt * sumY[j]) + (0.015) * q_m[j]);
                    }
                }
                
                for (int R_u_k: R_u) {
                    double * y_r_u_k = y[R_u_k];
                    for (int j = 0; j < lf; j++) {
                        y_r_u_k[j] -= (0.007) * decay * (gradY[j] + 0.015 * y_r_u_k[j]);
                    }
                }
            }
            
            decay *= 0.8;
        }
        
    }

    vector<double> predict(Dataset &data, Dataset &train_data) {
        vector<double> pred;
        for(int u = 0; u < n_users; u++) {
            vector<int> R_u = train_data.user_data[u].R_u;
            for(Example ex: data.user_data[u].entries) {
                pred.push_back(pred_one_test(u, ex.m, R_u));
            }
        }
        return pred;
    }

    double error(Dataset &data, Dataset &train_data) {
        vector<double> pred = predict(data, train_data);
        vector<double> trut;
        for(int u = 0; u < n_users; u++) {
            for(Example ex: data.user_data[u].entries) {
                trut.push_back(ex.r);
            }
        }
        double SE = 0.0;
        for(int i = 0; i < pred.size(); i++) {
            SE += pow(pred[i] - trut[i], 2);
        }
        return (pow(SE / pred.size(), 0.5));
    }
};

//function to compute average
double compute_average(vector<int> &vi) {
    
    double sum = 0;

    // iterate over all elements
    for (int p:vi){
     sum = sum + p;
    }

    if (vi.size() > 0) {
      return (sum/vi.size());
    }
    else {
      return 0.0;
    }
}

// r user, c movie, v rating
Dataset load_data(string path) {
    Dataset data;
    data.user_data.resize(n_users);

    io::CSVReader<3> in(path);
    in.read_header(io::ignore_extra_column, "User Number", "Movie Number", "Rating");
    
    double sum = 0.0;
    data.n_examples = 0;
    int u, m, r;
    while(in.read_row(u, m, r)) {
        Example ex = {m-1, r};
        data.user_data[u-1].entries.push_back(ex);
        data.user_data[u-1].R_u.push_back(m-1);
        sum += r;
        data.n_examples++;
    }
    data.mean = sum / data.n_examples;

    return data;
}

void save_submission(string model_name, string ordering, string source, vector<double> predictions) {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,sizeof(buffer),"%b%d%H%M%S", timeinfo);

    string path = "../submissions/" + ordering + "_" + model_name + "_" + source + "_" + buffer + ".pred";

    FILE * file = fopen(path.c_str(), "w");
    for (double p: predictions) {
        fprintf(file, "%.3f\n", p);
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    int latentFactors = 200;
    int epochs = 40;
    double lr = 0.007;   //
    double reg = 0.015; // these parameters are currently hardcoded

    cout << "Loading data..." << endl;

    Dataset train_set = load_data("../data/real_mu_train.csv");
    Dataset test_set1 = load_data("../data/real_um_probe_sorted.csv");
    Dataset test_set2 = load_data("../data/um_qual.csv");

    cout << "Training model..." << endl;
    cout << "Using "<< latentFactors <<" latent factors "<<endl;
    cout << "Epochs: "<< epochs << " learning rate: "<< lr <<" reg: "<< reg << endl;

    clock_t t = clock();

    SVDpp model(latentFactors);
    cout<<"Model initalized! model used to train um_train.csv"<<endl;
    model.train(train_set, test_set1, epochs);

    double t_delta = (double) (clock() - t) / CLOCKS_PER_SEC;

    printf("Training time: %.2f s\n", t_delta);
    
    // double rmse = model.error(train_set, train_set);
    // printf("Train RMSE: %.3f\n", rmse);
    double rmse = model.error(test_set1, train_set);
    printf("Val RMSE: %.3f\n", rmse);
    
    vector<double> predictions = model.predict(test_set1, train_set);
    save_submission("svd++", "um", "real_probe", predictions);
    predictions = model.predict(test_set2, train_set);
    save_submission("svd++", "um", "qual", predictions);
    
}
