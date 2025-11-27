import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { Point } from './Point';
import { refinements } from './config';
import { distance, centroid, dispose } from './utility';
import { GroupData } from './GroupData';
import { Logger } from './Logger';

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

async function getClusterPoints(
  clusterIndex: number,
  points: tf.Tensor,
  assignments: tf.Tensor
): Promise<tf.Tensor2D> {
  const clusterTensor = tf.tensor1d([clusterIndex]);
  // array of bools stating if a given array position is a member of the current group
  const boolTensor = tf.equal(clusterTensor, assignments);
  dispose(clusterTensor);
  Logger.trace(`Group membership state for group ${clusterIndex}`, boolTensor);
  // array of indices of points that are a member of the current group
  const boolIndexTensor = await tf.whereAsync(boolTensor);
  dispose(boolTensor);
  // The actual points that are a member of the current group
  const clusterPoints = points.gather(boolIndexTensor) as tf.Tensor2D;
  dispose(boolIndexTensor);
  return clusterPoints;
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
  let assignmentsIterated: number[] = [];
  let mean0Iterated: number[] = [];
  let mean1Iterated: number[] = [];
  const clusters = 2;
  for (let i = 0; i < refinements; i++) {
    const distances = getDistances(centroids, points);
    // assign group number (0 or 1) based on if point at that index is closer to
    //centroid 1 or centroid 2

    const assignments = distances.argMin(-1);
    Logger.trace('distances', distances);
    const means: Array<tf.Tensor> = [
      tf.tensor2d([initialCentroids[0].x, initialCentroids[0].y], [1, 2]),
      tf.tensor2d([initialCentroids[1].x, initialCentroids[1].y], [1, 2]),
    ];
    // Initialize with starting centerpoints, will always be size 2
    await (async () => {
      for (let clusterIndex = 0; clusterIndex < clusters; clusterIndex += 1) {
        const clusterPoints = await getClusterPoints(clusterIndex, points, assignments);
        // Find the new centeroid of those points in the current group
        if (clusterPoints.shape[0] > 1) {
          // Don't find centroid if group not defined
          const mean = centroid(clusterPoints);
          dispose(means[clusterIndex]);
          means[clusterIndex] = mean;
          Logger.trace('Cluster can be grouped', clusterPoints);
        } else {
          // mean[clusterIndex] remains with default initial centroid value;
          Logger.trace('Cluster cannot be grouped, using default', clusterPoints);
        }
        // We have an extra empty dimension here we don't need, so remove it with `squeeze`.
        const centroidsGroup = tf.stack([means[0], means[1]]);
        dispose(centroids);
        centroids = centroidsGroup.squeeze([1]); // save new centroids
        [centroidsGroup, clusterPoints].forEach((t: tf.Tensor | tf.Tensor2D) => dispose(t));
        assignmentsIterated = (await assignments.array()) as number[];
        const rehapedMean1 = means[0].reshape([2]);
        const rehapedMean2 = means[1].reshape([2]);

        mean0Iterated = (await rehapedMean1.array()) as number[]; // save centroids as array
        mean1Iterated = (await rehapedMean2.array()) as number[];

        dispose(rehapedMean1);
        dispose(rehapedMean2);
      }
      Logger.info(`Centroids, iteration ${i}`, centroids);
    })();

    [distances, assignments, means[0], means[1]].forEach((t: tf.Tensor | tf.Tensor2D) =>
      dispose(t)
    );
  }

  // You could derive these means from the single centroids tensor here rather than keeping the extra
  // arrays, but because because we already have means[0] and means[1],
  // are iterating on tensors, and getting data out of them
  // is an async operation, I found it easier to just store mean0Iterated and mean0Iterated.
  // Basically, the centroids tensor is used for the calculations, and the means array
  // more conveniently holds the output we pass to GroupData(...)
  dispose(centroids);

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
