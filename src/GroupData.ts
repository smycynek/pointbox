import { Point } from './Point';

/*
Contain all points, which group they belong to, and some methods to quickly
determine if one group has no points yet
*/
export class GroupData {
  public constructor(
    public assignments: number[],
    public centerPoints: Point[]
  ) {}
  // If a group has 0 or 1 points, don't render the group
  public group1IsDefault(): boolean {
    return this.assignments.filter((n: number) => n === 0).length < 2;
  }
  public group2IsDefault(): boolean {
    return this.assignments.filter((n: number) => n === 1).length < 2;
  }
}
