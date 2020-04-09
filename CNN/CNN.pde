String[][] result;
NeuralNetwork net;
double error;

void setup() {
  size(600, 500);

    ImageData idata= new ImageData(100, 100);
    idata.prepareTXT(5, 1, 2);

  double train_oran=0.95;
  int iteration=20000;
  Data data= new Data("data/data.txt");
  //  data.parameters_scale();
  data.splitData(train_oran);

  println("Column Size: "+(data.all_data[0].length-data.resultList.size()));
  println("Data is ready!");

  net = new NeuralNetwork(new int[]{ 16,16, data.resultList.size()}); 
  net.setFunctionType(ActF.SIGMOID);
  net.setLearningRatio(0.5);  

  println("Network is ready!");   
  net.train(data, iteration); 
  error=net.sqError;
  result = net.test(data);

}

void draw() {  

  background(255);
  tag("ACCURACY", result[result.length-1][1], 0, 0);
  tag("M.SQ.ERR", nf((float)net.iterErr[net.iterErr.length-1]), 300, 0);
  net.drawNetwork(0, 120);
  net.drawError(10,25,width-20,80);
}

void tag(String title, String message, int x, int y) {
  textSize(12);
  fill(0);
  rect(x, y, textWidth(title)+20, 20);
  text(message, x+textWidth(title)+30, y+15);
  fill(255, 0, 0);
  text(title, x+10, y+15);
}

void keyPressed() {
  if (keyCode==ENTER)
    println("\n"+net.getNetworkJSON());
}
