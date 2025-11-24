import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

import { Point } from './Point';
import { refinements } from './config';
import { distance, centroid } from './utility';
import { GroupData } from './GroupData';
import { Logger } from './Logger';

// Typically true, just for debugging
export function gcEnabled() {
  return true;
}

function dispose(tensor: tf.Tensor | tf.Tensor2D | null) {
  if (gcEnabled()) {
    if (tensor) {
      tensor.dispose();
    }
  }
}

export function getDistances(centroids: tf.Tensor, points: tf.Tensor) {
  return tf.tidy(() => {
    // We need to add extra array dimensions with `expandDims` in different places so that
    // the distance formula can work on a vector rather than a scalar for each centroid against
    // each point
    const expandedPoints = points.expandDims(1);
    const expandedCentroids = centroids.expandDims(0);
    return distance(expandedCentroids, expandedPoints); // find distance between each point and each centroid
  });
}

async function iterativeGroup(
  points: tf.Tensor2D,
  refinements: number,
  initialCentroids: Point[]
): Promise<GroupData> {
  // Starting with inital/default centroids, attempt to assign
  // points to groups associated with each centroid.
  const initialCentroidsRaw = [
    [initialCentroids[0].x, initialCentroids[0].y],
    [initialCentroids[1].y, initialCentroids[1].y],
  ];

  let centroids = tf.tensor(initialCentroidsRaw);
  let centroidsIntermediate: tf.Tensor | null = null;
  let assignmentsIterated: number[] = [];
  let mean0Iterated: number[] = [];
  let mean1Iterated: number[] = [];
  const clusters = 2;
  for (let i = 0; i < refinements; i++) {
    const distances = getDistances(centroids, points);
    // assign group number (0 or 1) based on if point at that index is closer to
    //centroid 1 or centroid 2

    Logger.info('distances', distances);
    const assignments = distances.argMin(-1);

    const means: Array<tf.Tensor> = [
      tf.tensor2d([initialCentroids[0].x, initialCentroids[0].y], [1, 2]),
      tf.tensor2d([initialCentroids[1].x, initialCentroids[1].y], [1, 2]),
    ];
    // Initialize with starting centerpoints, will always be size 2
    await (async () => {
      for (let clusterIndex = 0; clusterIndex < clusters; clusterIndex += 1) {
        const clusterTensor = tf.tensor1d([clusterIndex]);
        // array of bools stating if a given array position is a member of the current group
        const boolTensor = tf.equal(clusterTensor, assignments);
        Logger.info(`Group membership state for group ${clusterIndex}, iteration ${i}`, boolTensor);
        // array of indices of points that are a member of the current group
        const boolIndexTensor = await tf.whereAsync(boolTensor);

        // The actual points that are a member of the current group
        const clusterPoints = points.gather(boolIndexTensor);
        Logger.info(`Points for group ${clusterIndex}, iteration ${i}`, clusterPoints);
        // Find the new centeroid of those points in the current group
        if (clusterPoints.shape[0] > 1) {
          // Don't find centroid if group not defined
          const mean = centroid(clusterPoints);
          means[clusterIndex] = mean;
          Logger.info('Cluster can be grouped', clusterPoints);
        } else {
          // mean[clusterIndex] remains with default initial centroid value;
          Logger.info('Cluster cannot be grouped, using default', clusterPoints);
        }

        // We have an extra empty dimension here we don't need, so remove it with `squeeze`.
        centroidsIntermediate = tf.stack([means[0], means[1]]).squeeze([1]); // save new centroids
        [clusterPoints, boolIndexTensor, boolTensor, clusterTensor].forEach(
          (t: tf.Tensor | tf.Tensor2D) => dispose(t)
        );
        assignmentsIterated = (await assignments.array()) as number[];
        mean0Iterated = (await means[0].reshape([2]).array()) as number[]; // save centroids as array
        mean1Iterated = (await means[1].reshape([2]).array()) as number[];
      }
    })();

    if (centroidsIntermediate) {
      centroids = centroidsIntermediate;
    }
    [distances, assignments, means[0], means[1]].forEach((t: tf.Tensor | tf.Tensor2D) =>
      dispose(t)
    );
  }
  Logger.info('Centroids', centroids);

  // You could derive these means from the single centroids tensor here rather than keeping the extra
  // arrays, but because because we already have means[0] and means[1],
  // are iterating on tensors, and getting data out of them
  // is an async operation, I found it easier to just store mean0Iterated and mean0Iterated.
  // Basically, the centroids tensor is used for the calculations, and the means array
  // more conveniently holds the output we pass to GroupData(...)
  [centroids, centroidsIntermediate].forEach((t: tf.Tensor | tf.Tensor2D | null) => dispose(t));
  return new GroupData(assignmentsIterated, [
    new Point(mean0Iterated[0], mean0Iterated[1]),
    new Point(mean1Iterated[0], mean1Iterated[1]),
  ]);
}

// Called from UI, takes points from model and returns
// group/assignment data and centroid data.
export async function groupPointsFromArray(
  points: Point[],
  initialCentroids: Point[]
): Promise<GroupData> {
  const pointArrayData = points.map((p: Point) => p.toArray());
  const pointDataTensor = tf.tensor2d(pointArrayData);
  const groupData = await iterativeGroup(pointDataTensor, refinements, initialCentroids);
  dispose(pointDataTensor);
  return groupData;
}
