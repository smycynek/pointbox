import { round2 } from './utility';

const formatter = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2 });

export class Point {
  public constructor(
    public x: number,
    public y: number
  ) {}
  public toString(): string {
    return `(${formatter.format(round2(this.x)).padStart(6, '0')},
    ${formatter.format(round2(this.y)).padStart(6, '0')})`;
  }
  public toArray(): number[] {
    return [this.x, this.y];
  }
}
