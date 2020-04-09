class Sobel {

  //  public ArrayList<double[][]> matrices = new ArrayList<double[][]>();

  public double[][] kernel3 = {  {1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  public double[][] sobel_x3 = {  {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  public double[][] sobel_y3 = {  {-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  public double[][] sharpen = {  {-1, -1, -1}, {-1, -8, -1}, {-1, -1, -1}};


  Sobel() {
  }

  PImage kernel_filter(PImage img, double[][] k, int stride) {

    int k_size1= k.length;
    int k_size2 = k[0].length;

    double k_sum=0;
    for (int i=0; i<k_size1; i++)
      for (int j=0; j<k_size2; j++)
        k_sum+=k[i][j];

    PImage out = createImage((img.width-k_size1+1)/stride, (img.height-k_size2+1)/stride, RGB);

    for (int x=0; x<img.width; x+=stride) {
      for (int y=0; y<img.height; y+=stride) {      
        float r=0;
        float g=0;
        float b=0;
        for (int i=0; i<k_size1; i++) {
          for (int j=0; j<k_size2; j++) {
            r+=(red(img.get(x+i, y+j))*k[i][j]);
            g+=(green(img.get(x+i, y+j))*k[i][j]);
            b+=(blue(img.get(x+i, y+j))*k[i][j]);
          }
        }
        int valr = (int)r;
        int valg = (int)g;
        int valb = (int)b;

        if (k_sum!=0) {
          valr = (int)(r/k_sum);
          valg = (int)(g/k_sum);
          valb = (int)(b/k_sum);
        }

        if (valr<0)
          valr=-1*valr;
        if (valr>255)
          valr=255;

        if (valg<0)
          valg=-1*valg;
        if (valg>255)
          valg=255;

        if (valb<0)
          valb=-1*valb;
        if (valb>255)
          valb=255;

        color c = color(valr, valg, valb);
        out.set( (int)(x/stride+k_size1/2), (int)(y/stride+k_size2/2), c);
      }
    }
    return out;
  }

  PImage sobel_filter(PImage img, double[][] kx, double[][] ky, boolean isColored, boolean isReverse, int stride) {

    int k_size1= kx.length;
    int k_size2 = kx[0].length;

    double kx_sum=0;
    double ky_sum=0;
    for (int i=0; i<k_size1; i++)
      for (int j=0; j<k_size2; j++) {
        kx_sum+=kx[i][j];
        ky_sum+=ky[i][j];
      }


    PImage out = createImage((img.width-k_size1+1)/stride, (img.height-k_size2+1)/stride, RGB);

    for (int x=0; x<img.width; x+=stride) {
      for (int y=0; y<img.height; y+=stride) {    
        float[] colorx= {0, 0, 0, 0};
        float[] colory= {0, 0, 0, 0};
        float[] colort= {0, 0, 0, 0};
        for (int i=0; i<k_size1; i++) {
          for (int j=0; j<k_size2; j++) {
            if (isColored) {
              colorx[0]+=(red(img.get(x+i, y+j))*kx[i][j]);
              colorx[1]+=(green(img.get(x+i, y+j))*kx[i][j]);
              colorx[2]+=(blue(img.get(x+i, y+j))*kx[i][j]);

              colory[0]+=(red(img.get(x+i, y+j))*ky[i][j]);
              colory[1]+=(green(img.get(x+i, y+j))*ky[i][j]);
              colory[2]+=(blue(img.get(x+i, y+j))*ky[i][j]);
            } else {
              colorx[3]+=(brightness(img.get(x+i, y+j))*kx[i][j]);
              colory[3]+=(brightness(img.get(x+i, y+j))*ky[i][j]);
            }
          }
        }


        for (int i=0; i<colorx.length; i++) {
          if (kx_sum!=0) 
            colorx[i] = (int)(colorx[i]/kx_sum);
          if (ky_sum!=0) 
            colory[i] = (int)(colory[i]/ky_sum);

          colort[i]= (int)Math.sqrt(colorx[i]*colorx[i]+colory[i]*colory[i]);

          if (colort[i]>255)
            colort[i]=255;

          if (isReverse) {
            colort[i]=255-colort[i];
          }
        }
        //      printArray(colort);
        color c = color(colort[3]);
        if (isColored)
          c=color(colort[0], colort[1], colort[2]);
        out.set( (int)(x/stride+k_size1/2), (int)(y/stride+k_size2/2), c);
      }
    }
    return out;
  }


  PImage pooling(PImage img, int stride, boolean isMax) {

    PImage out = createImage(img.width/stride, img.height/stride, RGB);

    for (int x=0; x<img.width; x+=stride) {
      for (int y=0; y<img.height; y+=stride) {      
        float r=0;
        float g=0;
        float b=0;
        for (int i=0; i<stride; i++) {
          for (int j=0; j<stride; j++) {
            if (isMax) {
              r=max(red(img.get(x+i, y+j)), r);
              g=max(green(img.get(x+i, y+j)), r);
              b=max(blue(img.get(x+i, y+j)), r);
            } else {
              r+=red(img.get(x+i, y+j));
              g+=green(img.get(x+i, y+j));
              b+=blue(img.get(x+i, y+j));
            }
          }
        }
        if (!isMax) {
          r=r/(stride*stride);
          g=g/(stride*stride);
          b=b/(stride*stride);
        }

        int valr = (int)r;
        int valg = (int)g;
        int valb = (int)b;

        if (valr>255)
          valr=255;

        if (valg>255)
          valg=255;

        if (valb>255)
          valb=255;

        color c = color(valr, valg, valb);
        out.set( (int)(x/stride), (int)(y/stride), c);
      }
    }
    return out;
  }
}