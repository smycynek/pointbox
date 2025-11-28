/* eslint-disable @typescript-eslint/no-extraneous-class */

import * as tf from '@tensorflow/tfjs';

/*
I didn't feel like installing a logging package, so this is fine for now.
*/
export enum LoggerLevel {
  None,
  Info,
  Trace,
}

export class Logger {
  private constructor() {}
  public static loggerLevel: LoggerLevel = LoggerLevel.Info;
  static info(prefix: string, data: string | tf.Tensor | tf.Tensor2D) {
    if (Logger.loggerLevel !== LoggerLevel.None) {
      Logger.implementation(`INFO: ${prefix}`, data);
    }
  }

  static trace(prefix: string, data: string | tf.Tensor | tf.Tensor2D) {
    if (Logger.loggerLevel === LoggerLevel.Trace) {
      Logger.implementation(`TRACE: ${prefix}`, data);
    }
  }

  static implementation(prefix: string, data: string | tf.Tensor | tf.Tensor2D) {
    if (data instanceof tf.tensor2d) {
      console.log(prefix);
      data.print();
    } else if (data instanceof tf.Tensor) {
      console.log(prefix);
      data.print();
    } else {
      console.log(`${prefix} ${data}`);
    }
    console.log('---');
  }
}
