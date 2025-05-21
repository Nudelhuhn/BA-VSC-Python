/**
 * Beschreibt einen Punkt im zweidimensionalen Koordinatensystem mit x- und
 * y-Wert
 */
public class Point {
	
	// Koordinaten
	private double x;
	private double y;
	
	public Point(){
	}
	public Point(double initX, double initY) {
		//System.out.println("initX und initY");
x=initX;
y=initY;
	}

	public double getX() {
		//System.out.println("getX");
		return x;
	}

	public void setX(double newX) {
		//System.out.println("newX");
		x=newX;
	}

	public double getY() {
		//System.out.println("getY");
		return y;
	}

	public void setY(double newY) {
		//System.out.println("newY");
		y=newY;
	}

	public double getDistance(Point p) {
		double x1=this.x;
		double y1=this.y;
		//System.out.println("point p");
	
		return Math.sqrt((x1-p.getX())*(x1-p.getX())+(y1-getY())*(y1-getY()));
	}
}
