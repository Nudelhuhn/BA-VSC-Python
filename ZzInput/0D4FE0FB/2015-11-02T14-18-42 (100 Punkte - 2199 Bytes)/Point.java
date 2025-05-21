
public class Point {
	
	 // Koordinaten
	   private double x;
	   private double y;

	   public Point() {
	   }

	   public Point(double initX, double initY) {
	      this.x = initX;
	      this.y = initY;
	   }

	   public double getX() {
	      return x;
	   }

	   public void setX(double newX) {
	      this.x = newX;
	   }

	   public double getY() {
	      return y;
	   }

	   public void setY(double newY) {
	      this.y = newY;
	   }

	   public double getDistance(Point p) {
	      return Math.sqrt(Math.pow(getX() - p.getX(), 2) + (Math.pow(getY() - p.getY(), 2)));
	   }
	
	
	
	
	

}
