class NeuralNetwork { //<>//
  int[][] test_table;
  double sqError;
  double[] iterErr;
  Layer[] layers;
  double[] target;
  double[] inputs;
  double[] outputs;
  int functionType = ActF.SIGMOID;
  double LEARNING_RATIO=0.5;

  NeuralNetwork(Layer[] _layers) {
    layers=_layers;
  }

  NeuralNetwork(int[] neuronPerLayer) {
    layers=new Layer[neuronPerLayer.length];
    for (int i=0; i<layers.length; i++)
      layers[i]=new Layer(neuronPerLayer[i]);
  }

  void setFunctionType(int type) {
    functionType= type;
    for (int i=0; i<layers.length; i++)
      layers[i].functionType = functionType;
  }

  void setLearningRatio(double LR) {
    LEARNING_RATIO= LR;
    for (int i=0; i<layers.length; i++)
      layers[i].LEARNING_RATIO= LR;
  }

  void setInputs(double[] _inputs) {
    inputs=_inputs;
  }

  void setTarget(double[] _target) {
    target=_target;
    outputs= new double[target.length];
  }

  void forward() {
    layers[0].setInputs(inputs);
    for (int i=1; i<layers.length; i++) {
      layers[i].setInputs(layers[i-1].getOutput());
    }
    outputs =layers[layers.length-1].getOutput();
    sqError=0;
    for (int i=0; i<outputs.length; i++)
      sqError+= pow((float)(target[i]-outputs[i]), 2f);
    sqError=sqError/outputs.length;
  }

  void backward() {
    deltalarHesapla();
    for (int i=0; i < layers.length; i++)       
      for (int j=0; j<layers[i].neurons.length; j++)
        layers[i].neurons[j].updateWeight(layers[i].neurons[j].delta);
  }

  void deltalarHesapla() {
    for (int i=layers.length-1; i >= 0; i--) {
      if (i==layers.length-1) {
        for (int j=0; j<layers[i].neurons.length; j++) {
          layers[i].neurons[j].delta = (target[j] - layers[i].neurons[j].output) * 
            derivative(layers[i].neurons[j].output, functionType, layers[i].neurons[j].function_input);
        }
      } else {
        for (int j=0; j<layers[i].neurons.length; j++) {
          double errorSum=0;
          for (int k=0; k<layers[i+1].neurons.length; k++)
            errorSum += layers[i+1].neurons[k].delta * layers[i+1].neurons[k].weights[j];

          layers[i].neurons[j].delta = errorSum * derivative(layers[i].neurons[j].output, functionType, layers[i].neurons[j].function_input);
        }
      }
    }
  }

  void train(Data data, int iteration) {
    int outSize= data.resultList.size();
    double[] inp= new double[data.train_data[0].length-outSize];
    double[] tar=new double[outSize];
    iterErr=new double[iteration];
    int lastPart = 0;
 //   data.shuffleData(data.train_data);
    print("| Training process below! |\n =");
    for (int cnt=0; cnt<iteration; cnt++) {
      double err=0;
      for (int i=0; i<data.train_data.length; i++) {
        for (int j=0; j<data.train_data[0].length-outSize; j++) {
          inp[j]=data.train_data[i][j];
        }
        for (int j=0; j<outSize; j++) {
          tar[j]=data.train_data[i][data.train_data[0].length-outSize+j];
        }
        setInputs(inp);
        setTarget(tar);
        forward();
        backward(); 
        err+=sqError;
      }
      int part=cnt*25/iteration;
      if (part>lastPart) {
        print("=");
        lastPart=part;
      }
      iterErr[cnt]=err/data.train_data.length;
    }
    println("\nTraining completed");
  }

  String[][] test(Data data) {
    println("\nTest started...");
    int score=0;
    int resSize=data.resultList.size();
    double[] inp= new double[data.train_data[0].length-resSize];
 //   data.shuffleData(data.test_data);

    test_table = new int [resSize][resSize];
    for (int i=0; i<resSize; i++)
      for (int j=0; j<resSize; j++)
        test_table[i][j]=0;

    String[][] out = new String[data.test_data.length+1][2];
    double err=0;
    for (int i=0; i<data.test_data.length; i++) {
      inp= new double[data.test_data[0].length-resSize];
      for (int j=0; j<inp.length; j++) {
        inp[j]=data.test_data[i][j];
      }
      setInputs(inp); 
      forward();
      err+=sqError;

      int index=0;
      int gercekIndex=0;
      double maxRes = outputs[0];
      for (int j=0; j<outputs.length; j++) {
        if (outputs[j]>maxRes) {
          index=j;
          maxRes =  outputs[j];
        }
        if (data.test_data[i][(data.test_data[i].length-resSize+j)]==1)
          gercekIndex=j;
      }

      out[i][0]=data.resultList.get(index);
      out[i][1]=data.resultList.get(gercekIndex);
      test_table[gercekIndex][index]++;

      if (index==gercekIndex)
        score++;
    }
    err=err/data.test_data.length;
    out[out.length-1][0]="ACC";
    out[out.length-1][1]=(score*100)/data.test_data.length + "% ("+score+"/"+data.test_data.length+")";
    println( "Test Completed!\nACC: "+ (double)score/data.test_data.length + " ("+score+"/"+data.test_data.length+")\nTEST-MSE: "+err);

    print_test_result_table(data);
    return out;
  }

  void print_test_result_table(Data data) {
    print("\nR / P");
    int resSize=data.resultList.size();
    for (int i=0; i<resSize; i++)
      print("\t["+data.resultList.get(i)+"]");
    println("\tACC");
    for (int i=0; i<resSize; i++) {
      print("["+data.resultList.get(i)+"]\t");
      int tot=0;
      for (int j=0; j<resSize; j++) {
        print(test_table[i][j]+"\t");
        tot+=  test_table[i][j];
      }
      println(nf((float)test_table[i][i]/tot));
    }
  }

  String[][] predict(double[][] table_to_predict, Data trained_data) {    
    String[][] predictResults= new String[table_to_predict.length][table_to_predict[0].length+1];
    double[] inp= new double[table_to_predict[0].length];

    for (int i=0; i<table_to_predict.length; i++)
      for (int j=0; j<table_to_predict[0].length; j++) {
        predictResults[i][j] = String.format("%.2f", table_to_predict[i][j]);
        table_to_predict[i][j]= map((float)table_to_predict[i][j], (float) trained_data.maxMin[j][1], (float)trained_data.maxMin[j][0], 0, 1);
      }

    for (int i=0; i<table_to_predict.length; i++) {
      for (int j=0; j<inp.length; j++) {
        inp[j]=table_to_predict[i][j];
      }
      setInputs(inp); 
      forward();

      int index=0;
      double maxRes = outputs[0];
      for (int j=1; j<outputs.length; j++) {
        if (outputs[j]>maxRes) {
          index=j;
          maxRes =  outputs[j];
        }
      }
      predictResults[i][table_to_predict[0].length]=trained_data.resultList.get(index);
    }

    return predictResults;
  }

  String getNetworkJSON() {
    String json="{\"LR\":"+LEARNING_RATIO+",\"FUNC\":"+( functionType==0 ? "\"SIGMOID\"":"OTHER" )+",\"Layers\": [";
    for (int i=0; i<layers.length; i++) {
      json+="{ \"Level\":"+i+",\"Neurons\": [";
      for (int j=0; j<layers[i].neurons.length; j++) {
        json+= "{\"Index\": "+j+",\"Bias\":"+ layers[i].neurons[j].biasWeight+",\"Weights\": [";
        for (int k=0; k<layers[i].neurons[j].weights.length; k++) {
          json+= layers[i].neurons[j].weights[k];
          if (k<layers[i].neurons[j].weights.length-1)
            json+=",";
        }        
        json+="]}";
        if (j<layers[i].neurons.length-1)
          json+=",";
      }
      json+="]}";
      if (i<layers.length-1)
        json+=",";
    }
    json+="]}";
    return json;
  }

  void drawNetwork(int x, int y) {
    float minw=1000000;
    float maxw = -10000000;
    int maxn=0;
    float minb=minw;
    float maxb=maxw;
    for (int l=0; l<layers.length; l++) {
      if (layers[l].neurons.length>maxn)
        maxn=layers[l].neurons.length;
      for (int n=0; n<layers[l].neurons.length; n++) {
        float tempb=(float)layers[l].neurons[n].biasWeight;
        if (tempb>maxb)
          maxb=tempb;
        if (tempb<minb)
          minb=tempb;
        for (int w=0; w<layers[l].neurons[n].weights.length; w++) {
          float temp= (float)layers[l].neurons[n].weights[w];
          if (temp>maxw)
            maxw=temp;
          if (temp<minw)
            minw=temp;
        }
      }
    }
    int dx=(width-2*x)/(layers.length+2);
    int dy=(height-y)/maxn;
    int di=(height-y)/inputs.length;
    for (int l=0; l<layers.length; l++) {
      for (int n=0; n<layers[l].neurons.length; n++) {
        double cb = map((float)layers[l].neurons[n].biasWeight, minb, maxb, 0f, 255f);
        fill(0, 0, (int)cb);
        ellipse(x+dx*(l+2), y+dy*(n+(float)(maxn-layers[l].neurons.length)/2), 10, 10);   
        for (int w=0; w<layers[l].neurons[n].weights.length; w++) {
          double c = map((float)layers[l].neurons[n].weights[w], minw, maxw, 0f, 255f);
          stroke(color((int)c, 0, 0));
          strokeWeight(1);
          if (l>0) 
            line(x+dx*(l+1), y+dy*(w+(float)(maxn-layers[l-1].neurons.length)/2), x+dx*(l+2), y+dy*(n+(float)(maxn-layers[l].neurons.length)/2));
          else {
            fill(0, 255, 0);
            ellipse(x+dx*(l+1), y+di*w, 5, 5);
            line(x+dx*(l+1), y+di*w, x+dx*(l+2), y+dy*(n+(float)(maxn-layers[l].neurons.length)/2));
          }
        }
      }
    }
  }
  
  void drawError(int x,int y,int w,int h){
    h=h-10;
      double firstErr=iterErr[0];  
      float dx = (float)w/iterErr.length;
      fill(200);
      rect(x,y,w,h);
      fill(0);
      textSize(12);
      text("ERRORS BY ITERATION", x+(w-textWidth("ERRORS BY ITERATION"))/2, y+15);
      textSize(8);
      text(""+firstErr,x+1,y+10);
      text("0",x-textWidth("0")/2,y+h+10);
      text(""+iterErr.length,x+dx*iterErr.length-textWidth(""+iterErr.length)/2,y+h+10);
      stroke(0,0,255);
      for(int i=1;i<iterErr.length;i++){
        line(x+dx*(i-1),(float)(y+h-h/firstErr*iterErr[i-1]),x+dx*(i),(float)(y+h-h/firstErr*iterErr[i]));
        if(i%(iterErr.length/10)==0)
          text(""+i,x+dx*(i)-textWidth(""+i)/2,y+h+10);        
    }
  }
  
}