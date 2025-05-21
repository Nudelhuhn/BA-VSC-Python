/**
 * Beschreibt einen Punkt im zweidimensionalen Koordinatensystem mit x- und
 * y-Wert
 */
public class Point {

	// Koordinaten
	private double x;
	private double y;

	public Point() {
	}

	public Point(double initX, double initY) {
		x= initX;
		y= initY;
		

	}

	public double getX() {
		return x;
	}

	public void setX(double newX) {
		x = newX;
	}

	public double getY() {
		return y;
	}

	public void setY(double newY) {
		y= newY;
	}

	public double getDistance(Point p) {
		
		double d = 0;
	    Point p1 = new Point(x,y);
		d = Math.sqrt(Math.pow( p.getX()-p1.getX(), 2)+ Math.pow(p.getY()-p1.getY(), 2));
		return d;
		
	
	}
}
