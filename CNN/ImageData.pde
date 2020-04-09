import java.io.BufferedWriter;
import java.io.FileWriter;

class ImageData {

  StringList folderList = new StringList();
  int x=0, y=0;

  ImageData() {
    File[] files = listFiles("data");
    folderList = new StringList();
    for (int i = 0; i < files.length; i++) {
      if (files[i].isDirectory() && !files[i].getName().contains("out")){
        folderList.append(files[i].getName());
        println(files[i].getName());
      }
    }
  }

  ImageData(int _x, int _y) {
    x=_x;
    y=_y;
    File[] files = listFiles("data");
    folderList = new StringList();
    for (int i = 0; i < files.length; i++) {
      if (files[i].isDirectory() && !files[i].getName().contains("out")){
        folderList.append(files[i].getName());
        println(files[i].getName());
      }
    }
  }

  void prepareTXT(int stride, int pooling, int CN_iter) {
    deleteFile("out");
    deleteFile("data.txt");
    String str="";
    for (int i=0; i<folderList.size(); i++) {
      File[] files = listFiles("data/"+folderList.get(i));
      for (int j=0; j<files.length; j++) {
        println("data/"+folderList.get(i)+"/"+files[j].getName());
        PImage temp = loadImage("data/"+folderList.get(i)+"/"+files[j].getName());
        if (x*y>0) {
          temp.resize(x, y);
        }
        Convolution c = new Convolution();
        c.getConvResultList(temp, stride, pooling, CN_iter);
        str="";
        for (int k=0; k<c.out_list.size(); k++) {
          PImage img = c.out_list.get(k);          
          for (int x=0; x<img.width; x++) {
            for (int y=0; y<img.height; y++) {      
              float b=brightness(img.get(x, y))/255;
              str+=(red(img.get(x, y))/255+","+green(img.get(x, y))/255+","+blue(img.get(x, y))/255+",");
            }
          }
        }
        appendTextToFile("data.txt", str+folderList.get(i));
      }
    }
    println("OK => columnSize: "+split(str, ',').length+"\n");
  }

  void appendTextToFile(String filename, String text) {
    File f = new File(dataPath(filename));
    if (!f.exists()) {
      createFile(f);
    }
    try {
      PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(f, true)));
      out.println(text);
      out.close();
    }
    catch (IOException e) {
      e.printStackTrace();
    }
  }

  void createFile(File f) {
    File parentDir = f.getParentFile();
    try {
      parentDir.mkdirs(); 
      f.createNewFile();
    }
    catch(Exception e) {
      e.printStackTrace();
    }
  }

  void deleteFile(String fileName) {
    File f = new File(dataPath(fileName));
    if (f.exists()) {
      if(f.isDirectory()){
        File[] files = listFiles(dataPath(fileName));
//        println(files.length);
        for(int i=0;i<files.length;i++)
          files[i].delete();
      }
      f.delete();
      println(dataPath(fileName)+" silindi");
    }
  }
}