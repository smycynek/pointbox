import * as tf from '@tensorflow/tfjs';
import { Point } from './Point';
import '@tensorflow/tfjs-backend-webgpu'; // This adds the WebGL backend to the global backend registry
import { Logger } from './Logger';
import { getDistances } from './tensorGrouping';

/*
Typically true, just for debugging
*/
export function gcEnabled() {
  return true;
}

/*
Not strictly needed, just to make debugging easier
*/
export function dispose(tensor: tf.Tensor | tf.Tensor2D) {
  if (gcEnabled()) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const dataId: any = tensor.dataId;
    if (tensor.isDisposed) {
      Logger.trace(`Tensor id: ${dataId.id}`, 'Already disposed!');
    } else {
      Logger.trace(`Tensor id: ${dataId.id}`, 'Disposed!');
      tensor.dispose();
    }
  }
}

/*
Log tensor id, conveniently returns what is passed to it
*/
export function logId(
  tensor: tf.Tensor | tf.Tensor2D,
  prefix: string = ''
): tf.Tensor | tf.Tensor2D {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const dataId: any = tensor.dataId;
  Logger.trace(`${prefix} Tensor id:`, `${dataId.id}`);
  return tensor;
}

/*
Some implementations take random points from the collection of user-entered points, but
this also works well.
*/
export function randomSeedCentroid(max: number): Point {
  return new Point(Math.round(Math.random() * max), Math.round(Math.random() * max));
}

/*
From a collection of points, take the average xs and ys and return one
point (as a tensor)
*/
export function centroid(points: tf.Tensor2D | tf.Tensor): tf.Tensor {
  const centroid = points.mean(0);
  return logId(centroid, 'Centroid allocated');
}

/*
Return the distance of each point in pointList (the user points)
to each point in pointRefs (the center points) and return in a new tensor.
*/
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

/*
Most modern environments support webGPU. WebGL or CPU will also work
but will be slower
*/
export function enableBackEnd(): string {
  tf.setBackend('webgpu').then(() => {
    Logger.info('Backend set: ', tf.getBackend());
  });
  return tf.getBackend();
}

/*
Memory leaks in tensors are occasionally easy to miss, so this
helps us determine if anything was not properly freed.
*/
export function logMemory(label: string) {
  Logger.info(label, JSON.stringify(tf.memory(), null, 4));
}

/*
A quick demo of most of the functions used in this app.
*/
export async function functionDemo() {
  const work = async () => {
    Logger.info('-Demo-', 'Simpler demo of TensorFlowJS functioned used in this app.');
    Logger.info('-Example 1-', 'Distance to points');

    const centers = logId(
      tf.tensor2d([
        [1, 1],
        [10, 10],
      ])
    );
    Logger.info('Center points', centers);

    const points = logId(
      tf.tensor2d([
        [2, 2],
        [8, 8],
      ])
    );
    Logger.info('Points to measure', points);

    const distances = getDistances(centers, points);
    dispose(centers);
    dispose(points);
    Logger.info('Distances from each point to each center', distances);
    dispose(distances);
    Logger.info('-Example 1-', 'End');

    Logger.info('-Example 2-', 'Find lesser of each pair');
    const pairs = logId(
      tf.tensor2d([
        [1, 2],
        [5, 1],
        [3, 7],
      ])
    );
    Logger.info('a,b pairs', pairs);

    const pairMins = logId(pairs.argMin(-1));
    dispose(pairs);
    Logger.info('Min value of each pair', pairMins);
    dispose(pairMins);
    Logger.info('-Example 2-', 'End');
    Logger.info('-Example 3-', 'Group assignment');
    const originalData = logId(tf.tensor1d([0, 10, 20, 30, 40, 50, 60, 70]));
    const assignments = logId(tf.tensor1d(['a', 'a', 'a', 'b', 'b', 'a', 'a', 'a']));
    const dataMask = logId(tf.tensor1d(['b']));
    const dataMaskResult = logId(assignments.equal(dataMask));
    const indices = await tf.whereAsync(dataMaskResult);
    logId(indices);
    const filteredData = logId(tf.gather(originalData, indices), 'Filtered data allocated');
    Logger.info('Initial data', originalData);
    dispose(originalData);
    Logger.info('Group assignments', assignments);
    dispose(assignments);
    Logger.info('DataMask for choosing group b', dataMask);
    Logger.info('Data mask results for b', dataMaskResult);
    dispose(dataMask);
    dispose(dataMaskResult);
    Logger.info('Indices of group b items', indices);
    Logger.info('Group b items', filteredData);
    dispose(indices);
    dispose(filteredData);
    Logger.info('-Example 3-', 'End');
  };
  await work();
}
