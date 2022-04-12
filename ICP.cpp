#include "ICP.h"

void nearest_neighbour(const MatrixXd &V1, const MatrixXd &V2, MatrixXd &nn_V2){

    // return the nearest neighbour to V1 in V2 as nn_V2

    int n = V1.rows();

    //create params used for the octree
    std::vector<std::vector<int > > O_PI;
    MatrixXi O_CH;
    MatrixXd O_CN;
    VectorXd O_W;
    //create params used for the knn algorithm
    MatrixXi I;

    //sum of distances between neighbour points
    float s = 0;

    // Build octree with V2 points
    igl::octree(V2,O_PI,O_CH,O_CN,O_W);
    //Find nearest_neighbor of V1 points in V2 octree
    igl::knn(V1,1,O_PI,O_CH,O_CN,O_W,I);

    //fill nn_V2 with the matching points and compute the sum of neighbours distances
    for (int i=0 ; i<n ; i++){
        nn_V2.row(i) = V2.row(I(i));
        s += (nn_V2.row(i) - V1.row(i)).norm();
    }

    std::cout<<"The sum of the neighbours distances equals "<<s<<std::endl;
}


void transform(MatrixXd &V1,const MatrixXd &V2){

    //align V1 to V2 when V1 and V2 points are in correspondance

    //build mean matrices
    Vector3d Mx = V1.colwise().mean();
    Vector3d My = V2.colwise().mean();

    //substract the mean to each column of V1 and V2
    MatrixXd V1m = V1.rowwise() - Mx.transpose();
    MatrixXd V2m = V2.rowwise() - My.transpose();

    //build C
    MatrixXd C = (V2m.transpose())*V1m;

    //Compute the SVD of C
    JacobiSVD<MatrixXd> svd(C, ComputeFullU | ComputeFullV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();

    //Find the optimal Rotation
    MatrixXd Ropt;

    MatrixXd S = MatrixXd::Identity(3,3);
    S(2,2) = -1.;
    float det = (U*(V.transpose())).determinant();

    if (fabs(det-1.f) < 0.01f){
        Ropt = U*(V.transpose());
    }
    else{
        Ropt = U*S*(V.transpose());
    }

    //Deduct optimal translation
    MatrixXd topt = My - Ropt*(Mx);

    //Transform V1
    int n = V1.rows();
    for (int i=0 ; i<n ; i++){
        V1.row(i) = (Ropt*(V1.row(i).transpose()) + topt).transpose();
    }

}
