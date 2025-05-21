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

	public double getDistance(Point point) {
	double x1=x;
	double y1=y;
	
	
		System.out.println("point point");
	
		return Math.sqrt((x-x1)*(x-x1)+(y-y1)*(y-y1));
	}
}
