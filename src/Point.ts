import { round2 } from './utility';

const formatter = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2 });

/*
This class is simple enough that we barely need it, and we could enhance it further,
but it's mostly just to conveniently accept, display, and extract point data.
*/
export class Point {
  public constructor(
    public readonly x: number,
    public readonly y: number
  ) {}
  public toString(): string {
    return `(${formatter.format(round2(this.x)).padStart(6, '0')},
    ${formatter.format(round2(this.y)).padStart(6, '0')})`;
  }
  public toArray(): number[] {
    return [this.x, this.y];
  }
}
