#include "pca.h"


void k_nearest_neighbour(const MatrixXd &V1,Eigen::MatrixXi &I, int k){

    // return the k nearest neighbour index

    //create params used for the octree
    std::vector<std::vector<int > > O_PI;
    MatrixXi O_CH;
    MatrixXd O_CN;
    VectorXd O_W;

    // Build octree with V1 points
    igl::octree(V1,O_PI,O_CH,O_CN,O_W);
    //Find k nearest_neighbor of V1 points in V1 octree
    igl::knn(V1,k,O_PI,O_CH,O_CN,O_W,I);

}



void compute_normals(const MatrixXd &V1,const Eigen::MatrixXi &I, int k, MatrixXd &normals){
    // compute the normals using PCA

    MatrixXd X(k,3);
    MatrixXd C(k,3);
    MatrixXd M;
    MatrixXd U;
    MatrixXd V;
    float m;
    Vector3d N;
    float no;
    MatrixXd::Index minRow, minCol;

    int n = V1.rows();

    for (int i=0 ; i<n ; i++){

        //construction of X, the points cloud of the k nearest neighbours of the ith point of V1;
        for (int j=0 ; j<k ; j++){
            X.row(j) = V1.row(I(i,j));
        }

        //construction of M, the mean of the columns of X
        M = X.colwise().mean();
        //construction of C, the covariance matrix of X, C = X-M
        for (int j=0 ; j<k ; j++){
            C(j,0) = X(j,0) - M(0);
            C(j,1) = X(j,1) - M(1);
            C(j,2) = X(j,2) - M(2);
        }

        //Calculate the eigenvalues and eigenvectors of CTC
        EigenSolver<MatrixXd> es((C.transpose())*C);
        U = es.eigenvalues().real();
        V = es.eigenvectors().real();

        //Extract the index of the eigenvector of the minimum eigenvalue
        m = U.minCoeff(&minRow,&minCol);

        //Compute N
        N = V.col(minRow).transpose();
        no = N.norm();
        N = N/(10*no);

        normals.row(i) = N;


    }

  }
