import {Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{
  public x:tf.Tensor;
  public y:tf.Tensor;
  public z:tf.Tensor;
  public model:tf.Sequential;
  public taille:number=170;
  public poids:number=64;
  public output;
  modelCreated: boolean=false;
  modelTrained: boolean=false;
  learningRate=0.001;
  trainProgress: number=0;
  public classes = ['Maigreur', 'Normal', 'Surpoids', 'Obèse'];
  private currentLoss: number;

  constructor(){
  }
  ngOnInit(): void {
    this.x=tf.tensor([6,-8,9,4,8,-3,6,-3],[2,4]);
    this.y=tf.tensor2d([[6,-3,8],[4,3.3,1],[-6,3,-3],[5,7,-3]]);
  }

  onMultiplication() {
    this.z=this.x.matMul(this.y);
  }

  onTransposeX() {
    this.z=this.x.transpose();
  }
  onTransposeY() {
    this.z=this.y.transpose();
  }
  reluX() {
    this.z= this.x.relu()
  }

  sigmoid() {
    this.z=tf.sigmoid(this.y);
  }

  argMaxX() {
    //let argMax=this.x.argMax(0).arraySync();
    //this.z=tf.tensor2d([argMax]);
  }
  argMaxY() {
    //let argMax=this.y.argMax(1).arraySync();
    //this.z=tf.tensor2d([argMax]);
  }

  createModel(){
    this.model=tf.sequential();
    this.model.add(tf.layers.dense({
      units:8,
      inputShape:2,
      activation:'sigmoid'
    }));

    this.model.add(tf.layers.dense({
      units:4,
      activation:'softmax'
    }))

    this.model.compile({
      optimizer:tf.train.adam(this.learningRate),
      loss:tf.losses.meanSquaredError
    });
    this.modelCreated=true;

  }
  async trainModel(){
    let data=this.getDataSet();
    let xs=tf.tensor2d(data.input);
    let ys=tf.tensor2d(data.output);
    xs.print();
    ys.print();
    let epochs=500;
    let epochIteration=0;
      let result=await this.model.fit(xs,ys,{
        epochs:epochs,
        batchSize:30,
        callbacks: {
          onEpochEnd : (epoch, logs)=>{
            ++epochIteration;
            this.trainProgress=epochIteration*100/epochs;
            this.currentLoss=logs.loss;
            //console.log(epoch+"=> : "+logs.loss);
          },
          onBatchEnd:(batch,logs)=>{
            //console.log(batch+"=>"+logs.loss);
          }
        }
      });
    this.modelTrained=true;
  }

  getDataSet(){
    let input=[];
    let output=[];
    for (let p = 30; p <120 ; p+=10) {
      for (let t = 100; t <200 ; t+=10) {
        input.push([p,t]);
        output.push(this.getIMCClass(p,t));
      }
    }
    return {input:input,output:output};
  }

  getIMCClass(poids, taille){
    let imc=poids/((taille/100)*(taille/100));
    if(imc<18.5) return [1,0,0,0];
    else if(imc>=18.5 && imc<25) return [0,1,0,0];
    else if(imc>=25 && imc<30) return [0,0,1,0];
    else if(imc>=30) return [0,0,0,1];
  }

  predict() {
    console.log(this.taille);
    console.log(this.poids);
    let xs=tf.tensor2d([[parseFloat(this.poids),parseFloat(this.taille)]]);
    let predicted:tf.Tensor=this.model.predict(xs);
    let index=predicted.argMax(-1).arraySync()[0];
    this.output=this.classes[index];
  }

  saveModel() {
    this.model.save('localstorage://ImcModel')
      .then(result=>{
        alert("Success Model Saved");
      }, err=>{
        alert('Error saving model');
      });
  }

  loadModel() {
    tf.loadLayersModel('localstorage://ImcModel')
      .then((m)=>{
      this.model=m;
      this.model.compile({
        loss:tf.losses.meanSquaredError,
        optimizer:tf.train.adam(this.learningRate)
      })
      alert("Success loading Model");
      this.modelCreated=true;

      this.modelTrained=true;
    },err=>{
      alert('Error loading model');
    });
  }

  getCorrectOutput(){
    let c=this.getIMCClass(this.poids,this.taille);
    console.log(c);
    const t:tf.Tensor1D=tf.tensor1d(c);
    let maxIndex=t.argMax(-1).dataSync()[0];
    console.log(maxIndex);
    return this.classes[maxIndex];
  }

  async createCNNModel(){
    // Create the model
    let cnnModel=tf.sequential();
    // Add the First Convolution Layer
    cnnModel.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
    }));
   //Adding The Second Layer : MaxPooling Layer
    cnnModel.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    }));
    // Adding Another Convolutional Layer
    cnnModel.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
    }));
 //Adding Another MaxPooling Layer
    cnnModel.add(tf.layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    }));
  // Adding A Flatten Layer
    cnnModel.add(tf.layers.flatten());
   // Adding A Dense Layer (Fully Connected Layer) For Performing The Final Classification
    cnnModel.add(tf.layers.dense({
      units: 10,
      kernelInitializer: 'VarianceScaling',
      activation: 'softmax'
    }));
 // Compiling The Model
    cnnModel.compile({
      optimizer: tf.train.sgd(0.15),
      loss: 'categoricalCrossentropy'
    });

    // Train the model
    let images=[[]];
    let labels=[[]];
    let xs=tf.tensor2d(images);
    let ys=tf.tensor2d(labels)

    await cnnModel.fit(xs, labels, {
      batchSize: 60,
      epochs: 1
    });
    // Prediction

    let inputImage=[[]];
    let outputLabel=cnnModel.predict(inputImage);

    console.log(outputLabel);
  }
}
