import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { Point } from './Point';
import { refinements } from './config';
import { distance, centroid, dispose, logId } from './utility';
import { GroupData } from './GroupData';
import { Logger } from './Logger';

/*
Get distances from each user point to each centerpoint.
*/
export function getDistances(centroids: tf.Tensor, points: tf.Tensor) {
  // Note that the distance tensor allocated will need to be deallocated, but everything inside
  // the 'tidy' will be deallocated for you.
  return tf.tidy(() => {
    // We need to add extra array dimensions with `expandDims` in different places so that
    // the distance formula can work on a vector rather than a scalar for each centroid against
    // each point
    const expandedPoints = points.expandDims(1);
    const expandedCentroids = centroids.expandDims(0);
    return logId(distance(expandedCentroids, expandedPoints), 'Distances allocated'); // find distance between each point and each centroid
  });
}

/*
Given the a cluster 0 or 1 and the assignments array, retrieve points belonging to that
cluster.
*/
async function getClusterPoints(
  clusterIndex: number,
  points: tf.Tensor,
  assignments: tf.Tensor
): Promise<tf.Tensor2D | tf.Tensor> {
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
  return logId(clusterPoints, 'ClusterPoints allocated');
}

/*
Perform a k-means grouping of points, starting with arbitrary centerpoint and
iterating a few times (see config.ts)
*/
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

  let centroids = logId(tf.tensor(initialCentroidsRaw), 'Centroids allocated');
  let assignmentsIterated: number[] = [];
  const clusters = 2;
  for (let i = 0; i < refinements; i++) {
    // 1 -- Get distances to centers (centroids adjusted each iteration)
    const distances = getDistances(centroids, points);

    // 2 -- Assign group number (0 or 1) based on if point at that index is closer to
    // centroid 1 or centroid 2
    const assignments = distances.argMin(1);
    Logger.trace('distances', distances);
    const means: Array<tf.Tensor> = [
      tf.tensor2d([initialCentroids[0].x, initialCentroids[0].y], [1, 2]),
      tf.tensor2d([initialCentroids[1].x, initialCentroids[1].y], [1, 2]),
    ];
    // Initialize with starting centerpoints, will always be size 2
    await (async () => {
      for (let clusterIndex = 0; clusterIndex < clusters; clusterIndex += 1) {
        // 3 -- Get points assigned to a group
        const clusterPoints = await getClusterPoints(clusterIndex, points, assignments);
        // 4a -- Find the new centroid of those points in the current group
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
        // 4b -- update the centroid for use in the next iteration
        centroids = centroidsGroup.squeeze([1]); // save new centroids
        [centroidsGroup, clusterPoints].forEach((t: tf.Tensor | tf.Tensor2D) => dispose(t));
        assignmentsIterated = (await assignments.array()) as number[];
      }
      Logger.info(`Centroids, iteration ${i}`, centroids);
    })();

    [distances, assignments, means[0], means[1]].forEach((t: tf.Tensor | tf.Tensor2D) =>
      dispose(t)
    );
  }
  const meanRawData: number[][] = (await centroids.array()) as number[][];
  dispose(centroids);
  // 4c -- Return the new centerpoint, and use it as the starting centerpoint next time a point is added
  return new GroupData(assignmentsIterated, [
    new Point(meanRawData[0][0], meanRawData[0][1]),
    new Point(meanRawData[1][0], meanRawData[1][1]),
  ]);
}

/*
Called from UI, takes points from model and returns
group/assignment data and centroid data.
*/
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
