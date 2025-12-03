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

  static warn(prefix: string, data?: string | tf.Tensor | tf.Tensor2D) {
    Logger.implementation(prefix, LoggerLevel.Info, data);
  }

  static info(prefix: string, data?: string | tf.Tensor | tf.Tensor2D) {
    Logger.implementation(prefix, LoggerLevel.Info, data);
  }

  static trace(prefix: string, data?: string | tf.Tensor | tf.Tensor2D) {
    Logger.implementation(prefix, LoggerLevel.Trace, data);
  }

  static implementation(
    prefix: string,
    level: LoggerLevel,
    data?: string | tf.Tensor | tf.Tensor2D
  ) {
    if (Logger.loggerLevel === LoggerLevel.None) {
      return;
    }
    if (level === LoggerLevel.Trace && Logger.loggerLevel !== LoggerLevel.Trace) {
      return;
    }
    const logType = LoggerLevel[level].toString();
    if (data instanceof tf.tensor2d) {
      console.log(`${logType} - ${prefix}`);
      data.print();
    } else if (data instanceof tf.Tensor) {
      console.log(`${logType} - ${prefix}`);
      data.print();
    } else {
      console.log(`${logType} - ${prefix} ${data ?? ''}`);
    }
    console.log('---');
  }
}
