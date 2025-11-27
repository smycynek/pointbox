/* eslint-disable @typescript-eslint/no-extraneous-class */

import * as tf from '@tensorflow/tfjs';

// I didn't feel like installing a logging package, so this is fine for now.

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
    console.log(prefix);
    if (data instanceof tf.tensor2d) {
      data.print();
    } else if (data instanceof tf.Tensor) {
      data.print();
    } else {
      console.log(data);
    }
    console.log('---');
  }
}
