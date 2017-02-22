#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

RNG rng(12345);

int main(){

    Mat imagem = imread("1.jpg"), gray;
    //redimensionamento
    resize(imagem,imagem,Size(637, 877));


    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    int gabarito[10][5];
    int indices[4];

    //conversão para escala de cinza para uso do thresold adaptativo
    cvtColor(imagem, gray, CV_BGR2GRAY);

    //blur para remoção de ruídos do scanner
    medianBlur(gray, gray, 7);

    //thresold
    cv::adaptiveThreshold(gray, gray, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, -3);


    //pré-processamento: fechamento
    Mat elem1 = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 3, 3 ));
    cv::morphologyEx(gray,gray,cv::MORPH_DILATE,elem1,cv::Point( -1, -1 ));
    cv::morphologyEx(gray,gray,cv::MORPH_ERODE,elem1,cv::Point( -1, -1 ));

    //visão do threshold
    namedWindow("", WINDOW_NORMAL);
    imshow("", gray);

    //busca por contornos
    findContours(gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    /// Get the moments
    vector<Moments> mu(contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    { mu[i] = moments( contours[i], false ); }


    ///  Get the mass centers:
    vector<Point2f> mc( contours.size());
    vector<Point2f> pontos(4);

    /// zera o vetor
    for(int i = 0; i < 4; i++){
        pontos[i].x = -1;
        pontos[i].y = -1;
    }

    for( size_t i = 0; i < contours.size(); i++ ){
        //calcula os centros de massa de todas as regiões
        mc[i] = Point2f( static_cast<float>(mu[i].m10/mu[i].m00) , static_cast<float>(mu[i].m01/mu[i].m00) );

        //verifica a área da região que tá sendo tratada
        int m = contourArea(contours[i]);

        /*
         * Verifica se a região tratada é uma das regiões que nós procuramos,
         * tomando como o base o tamanho da região (verificando se ela está no
         * nosso intervale de interesse e posteriormente se ela está em um dos
         * quadrantes extremos. Esses quadrantes extremos foram gerados dividin-
         * do as linhas da imagem em 7 regiões e as colunas em 5.
         *
         * Desvio padrão da área de interesse calculado =~ 23
         * min 261
         * max 335.5
         * total 13 regiões
         *
        */
        if(m >= 238 && m <= 355){

            if(mc[i].x >= 0 && mc[i].x <=127 && mc[i].y >= 0 && mc[i].y <= 124)
            {
                if(pontos[0].x == -1){
                    pontos[0]=mc[i];
                    indices[0] = i;
                }
                else {
                    if(mc[i].y < pontos[0].y)
                    {
                        pontos[0] = mc[i];
                        indices[0] = i;
                    }
                    else if (mc[i].y == pontos[0].y)
                        if(mc[i].x < pontos[0].x){
                            pontos[0] = mc[i];
                            indices[0] = i;
                        }
                }
            }

            else if(mc[i].x >= 508 && mc[i].x <=636 && mc[i].y >= 0 && mc[i].y <= 124)
            {
                if(pontos[1].x == -1){
                    pontos[1]=mc[i];
                    indices[1]=i;
                }
                else {
                    if(mc[i].x > pontos[1].x){
                        pontos[1] = mc[i];
                        indices[1] = i;
                    }
                    else if(mc[i].x == pontos[1].x){
                        if(mc[i].y < pontos[1].y){
                            pontos[1] = mc[i];
                            indices[1] = i;
                        }
                    }
                }
            }

            else if(mc[i].x >= 0 && mc[i].x <=127 && mc[i].y >= 750 && mc[i].y <= 876)
            {
                if(pontos[2].x == -1){
                    pontos[2]=mc[i];
                    indices[2] = i;
                }
                else {
                    if(mc[i].y > pontos[2].y)
                    {
                        pontos[2] = mc[i];
                        indices[2] = i;
                    }
                    else if (mc[i].y == pontos[2].y)
                        if(mc[i].x < pontos[2].x){
                            pontos[2] = mc[i];
                            indices[2] = i;
                        }
                }
            }

            else if(mc[i].x >= 508 && mc[i].x <=636 && mc[i].y >= 750 && mc[i].y <= 876)
            {
                if(pontos[3].x == -1){
                    pontos[3]= mc[i];
                    indices[3] = i;
                }
                else{
                    if(mc[i].x > pontos[3].x){
                        pontos[3] = mc[i];
                        indices[3] = i;
                    }
                    else if(mc[i].x == pontos[3].x){
                        if(mc[i].y > pontos[3].y){
                            pontos[3] = mc[i];
                            indices[3] = i;
                        }
                    }
                }
            }


        }
    }


    std::cout << "Page markers identificados:"<< std::endl;
    std::cout << "\tPoint(x,y)=" << pontos[0] << std::endl;
    std::cout << "\tPoint(x,y)=" << pontos[1] << std::endl;
    std::cout << "\tPoint(x,y)=" << pontos[2] << std::endl;
    std::cout << "\tPoint(x,y)=" << pontos[3] << std::endl;

    /**
     * Mostrar na imagem as regiões extremas identificadas para correção da perscpectiva
     */
    Mat drawing = Mat::zeros(gray.size(), CV_8UC3 );
    for( size_t i = 0; i< 4; i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, (int)indices[i], color, 2, 8, hierarchy, 0, Point() );
        circle( drawing, mc[indices[i]], 4, color, -1, 8, 0 );
    }

    //definição dos pontos destino da transformação de perspectiva
    Mat target(imagem.rows, imagem.cols, imagem.type());
    vector<Point2f> target_points(4);
    target_points[0].x = 0; target_points[0].y = 0;
    target_points[1].x = imagem.cols - 1; target_points[1].y = 0;
    target_points[3].x = imagem.cols - 1;target_points[3].y = imagem.rows -1;
    target_points[2].x = 0; target_points[2].y = imagem.rows -1;

    //transformação de perscectiva
    Mat trans_mat = getPerspectiveTransform(pontos, target_points);
    warpPerspective(imagem, target, trans_mat, target.size());

    namedWindow( "Target 1", WINDOW_NORMAL );
    imshow( "Target 1", target );
    Mat final;

    //pré-processamento da imagem e conversão dela pra binário
    cvtColor(target, target, CV_BGR2GRAY);
    medianBlur(target, target, 11);
    threshold(target, final, 130, 255, THRESH_BINARY);
    bitwise_not(final, final);


    int backup = 48;
    int x = backup;
    int y = 61;


    //cálculo e verificação das áreas de interesse
    vector<Point> temp (4);
    for(int k = 0; k < 10; k++){
        for(int m = 0; m < 5; m++){
            temp[0].x = x;
            temp[0].y = y;

            temp[1].x = temp[0].x + 24;
            temp[1].y = y;

            temp[2].x = temp[0].x;
            temp[2].y = temp[0].y + 12;

            temp[3].x = temp[1].x;
            temp[3].y = temp[2].y;


            int bin[2] = {0, 0};
            for(int t = x; t <= temp[1].x; t++){
                for(int u = y; u <= temp[2].y; u++){
                    if(final.at<uchar>(u, t) == 0)
                        bin[0]++;
                    else
                        bin[1]++;
                }
            }
            if(bin [0] > bin[1]) gabarito[k][m] = 0;
            else gabarito[k][m] = 1;

            x = temp[1].x + 7;
        }
        y = temp[2].y + 10;
        x = backup;
    }

    //impressão do gabarito na tela

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "representação encontrada para o gabarito:" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    for(int i = 0; i < 10; i++){
        for(int k = 0; k < 5; k++){
            std::cout << "\t" << gabarito[i][k]<< "\t";
        }
        std::cout << endl;
    }



    std::cout << endl;
    std::cout << endl;
    std::cout << endl;
    std::cout << "\t\tRESULTADO" << endl;
    int veri;
    for(int i = 0; i < 10; i++){
        char c;
        veri = 0;
        std::cout << "RESPOSTA " << i +1 << "ª QUESTÃO: ";

        for(int k = 0; k < 5; k++){
            if(gabarito[i][k] == 1){
                if(k == 0) {
                    c = 'A';
                    veri++;
                }
                if(k == 1) {
                    c = 'B';
                    veri++;
                }
                if (k == 2) {
                    c = 'C';
                    veri++;
                }
                if (k == 3) {
                    c = 'D';
                    veri++;
                }
                if(k == 4) {
                    c = 'E';
                    veri++;
                }


            }

            if(k == 4){
                if( veri == 0){
                    std::cout << "QUESTÃO EM BRANCO" << std::endl;
                }
                else if (veri == 1){
                    std::cout << c << std::endl;
                }
                else{
                    std::cout << "QUESTÃO ANULADA, 2+ ALTERNATIVAS MARCADAS" << std::endl;
                }
            }
        }
    }


    namedWindow( "Contours", WINDOW_NORMAL );
    imshow( "Contours", drawing );

    namedWindow( "targ", WINDOW_NORMAL );
    imshow( "targ", target);

    imwrite("targ.jpg", target);

    namedWindow("result", WINDOW_NORMAL);
    imshow("result", final);
    waitKey();

    waitKey(0);
    return 1;
}
