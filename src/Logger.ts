/* eslint-disable @typescript-eslint/no-extraneous-class */

import * as tf from '@tensorflow/tfjs';

export class Logger {
  private constructor() {}
  public static enabled: boolean = false;
  static info(prefix: string, data: string | tf.Tensor | tf.Tensor2D) {
    if (Logger.enabled) {
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
}
