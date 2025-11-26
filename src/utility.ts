import * as tf from '@tensorflow/tfjs';
import { Point } from './Point';
import '@tensorflow/tfjs-backend-webgpu'; // This adds the WebGL backend to the global backend registry
import { Logger } from './Logger';
import { getDistances } from './tensorGrouping';

// Typically true, just for debugging
export function gcEnabled() {
  return true;
}

export function dispose(tensor: tf.Tensor | tf.Tensor2D | null) {
  if (gcEnabled()) {
    if (tensor) {
      // console.log('dispose');
      tensor.dispose();
    } else {
      console.log('Null tensor!');
    }
  }
}

export function randomSeedCentroid(max: number): Point {
  return new Point(Math.round(Math.random() * max), Math.round(Math.random() * max));
}

export function centroid(points: tf.Tensor2D): tf.Tensor {
  return points.mean(0);
}

export function distance(pointRefs: tf.Tensor, pointList: tf.Tensor): tf.Tensor {
  return tf.tidy(() => pointList.sub(pointRefs).square().sum(-1).sqrt());
}

export function getMousePos(canvas: HTMLCanvasElement, mouseEvent: MouseEvent): Point {
  const canvasRect: DOMRect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / canvasRect.width; // scale because of bitmapping
  const scaleY = canvas.height / canvasRect.height;
  return new Point(
    (mouseEvent.clientX - canvasRect.left) * scaleX,
    (mouseEvent.clientY - canvasRect.top) * scaleY
  );
}

export function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

export function enableBackEnd(): string {
  tf.setBackend('webgpu').then(() => {
    console.log('webgpu enabled');
  });
  return tf.getBackend();
}

export function logMemory(label?: string) {
  if (label) {
    console.log(label);
  }
  console.log(tf.memory());
}

export async function functionDemo() {
  Logger.enabled = true;
  const work = async () => {
    console.log('Simpler demo of TensorFlowJS functioned used in this app.');
    console.log('Example 1, distance to points');

    const centers = tf.tensor2d([
      [1, 1],
      [10, 10],
    ]);
    Logger.info('Center points', centers);

    const points = tf.tensor2d([
      [2, 2],
      [8, 8],
    ]);
    Logger.info('Points to measure', points);

    const distances = getDistances(centers, points);
    Logger.info('Distances from each point to each center', distances);
    console.log('------');

    console.log('Example 2, find lesser of each pair');
    const pairs = tf.tensor2d([
      [1, 2],
      [5, 1],
      [3, 7],
    ]);
    Logger.info('a,b pairs', pairs);

    const pairMins = pairs.argMin(-1);
    Logger.info('Min value of each pair', pairMins);
    console.log('-----');
    console.log('Example 3, group assignment');
    const originalData = tf.tensor1d([0, 10, 20, 30, 40, 50, 60, 70]);
    const assignments = tf.tensor1d(['a', 'a', 'a', 'b', 'b', 'a', 'a', 'a']);
    const dataMask = tf.tensor1d(['b']);
    const dataMaskResult = assignments.equal(dataMask);
    const indices = await tf.whereAsync(dataMaskResult);
    const filteredData = tf.gather(originalData, indices);
    Logger.info('Intial data', originalData);
    Logger.info('Group assignments', assignments);
    Logger.info('DataMask for choosing group b', dataMask);
    Logger.info('Data mask results for b', dataMaskResult);
    Logger.info('Indices of group b items', indices);
    Logger.info('Group b items', filteredData);
    console.log('-----');
    dispose(centers);
    dispose(points);
    dispose(distances);
    dispose(pairs);
    dispose(pairMins);
    dispose(originalData);
    dispose(assignments);
    dispose(dataMask);
    dispose(dataMaskResult);
    dispose(indices);
    dispose(filteredData);
  };
  await work();
}
