
public class Point {

	
	private double x;
	private double y;

	public Point() {
	}

	public Point(double initX, double initY) {
		x=initX;
		y=initY;

	}

	public double getX() {
		return x;
	
	}

	public void setX(double newX) {
		x=newX;
	}

	public double getY() {
		return -1.0;
	}

	public void setY(double newY) {
		y=newY;
	}
	

	public double getDistance(Point p) {
		double distanceX=Math.pow(p.getX()-x, 2);
		double distanceY=Math.pow(p.getY()-y,2);
		
		return -1.0;
	}


	
	}

