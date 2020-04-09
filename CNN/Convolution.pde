class Convolution {
  ArrayList<PImage> out_list = new ArrayList<PImage>();

  Convolution() {
  }

  ArrayList<PImage> getConvResultList(PImage image, int stride, int pooling, int CN_iter) {
    Sobel s = new Sobel();
    out_list = new ArrayList<PImage>();

    PImage temp_sbl=image;
    PImage temp_shr=image;
    for (int j=0; j<CN_iter; j++) {
      temp_shr=s.kernel_filter(temp_shr, s.sharpen, stride);
      temp_sbl =s.sobel_filter(temp_sbl,s.sobel_x3,s.sobel_y3,true,false,stride);
      
      temp_shr= s.pooling(temp_shr, pooling,false);
      temp_sbl= s.pooling(temp_sbl, pooling,false);
    }

 //   temp_shr.save("data/out/"+year()+"_"+month()+"_"+day()+"_"+hour()+"_"+minute()+"_"+second()+"_"+millis()+".jpg");
    out_list.add(temp_shr);

 //   temp_sbl.save("data/out/"+year()+"_"+month()+"_"+day()+"_"+hour()+"_"+minute()+"_"+second()+"_"+millis()+".jpg");
    out_list.add(temp_sbl);
  //  out_list.add(image);
    return out_list;
  }
}
