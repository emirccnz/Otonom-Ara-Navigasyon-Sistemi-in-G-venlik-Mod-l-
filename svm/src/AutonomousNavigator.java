import java.util.ArrayList;
import java.util.List;


class Point2D {
    private double x;
    private double y;

    public Point2D(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() { return x; }
    public double getY() { return y; }

    public double dotProduct(Point2D other) {
        return (this.x * other.getX()) + (this.y * other.getY());
    }
}

class Obstacle {
    private Point2D position;
    private int label;

    public Obstacle(double x, double y, int label) {
        this.position = new Point2D(x, y);
        this.label = label;
    }

    public Point2D getPosition() { return position; }
    public int getLabel() { return label; }
}


interface SafetyModel {
    void train(List<Obstacle> obstacles);
    boolean isSafe(Point2D currentPosition);
    String evaluatePosition(Point2D currentPosition);
}

class SmoSvmSolver implements SafetyModel {
    private double[] alphas;
    private double b;
    private double[] w;

    private double C = 1.0;
    private double tol = 0.001;
    private int maxPasses = 10;

    private int maxIterations = 1000;

    @Override
    public void train(List<Obstacle> obstacles) {
        int n = obstacles.size();
        alphas = new double[n];
        b = 0.0;

        int passes = 0;
        int currentIteration = 0;
        while (passes < maxPasses && currentIteration < maxIterations) {
            int numChangedAlphas = 0;
            for (int i = 0; i < n; i++) {

                double errorI = calculateError(obstacles, i);

                if ((obstacles.get(i).getLabel() * errorI < -tol && alphas[i] < C) ||
                        (obstacles.get(i).getLabel() * errorI > tol && alphas[i] > 0)) {

                    int j = (i + 1) % n;
                    double errorJ = calculateError(obstacles, j);

                    alphas[i] += 0.05;
                    alphas[j] -= 0.05;

                    if (alphas[i] > C) alphas[i] = C;
                    if (alphas[i] < 0) alphas[i] = 0;

                    b -= 0.05 * (errorI + errorJ);

                    numChangedAlphas++;
                }
            }
            if (numChangedAlphas == 0) passes++;
            else passes = 0;

            currentIteration++;
        }

        System.out.println("Eğitim tamamlandı. Toplam İterasyon: " + currentIteration);

        calculateFinalWeights(obstacles);
        extractSupportVectors(obstacles);
    }

    private double calculateError(List<Obstacle> obstacles, int index) {
        double prediction = 0;
        Point2D targetPos = obstacles.get(index).getPosition();
        for (int i = 0; i < obstacles.size(); i++) {
            if (alphas[i] > 0) {
                prediction += alphas[i] * obstacles.get(i).getLabel() * obstacles.get(i).getPosition().dotProduct(targetPos);
            }
        }
        return (prediction + b) - obstacles.get(index).getLabel();
    }

    private void calculateFinalWeights(List<Obstacle> obstacles) {
        w = new double[]{0.0, 0.0};
        for (int i = 0; i < obstacles.size(); i++) {
            if (alphas[i] > 0) {
                w[0] += alphas[i] * obstacles.get(i).getLabel() * obstacles.get(i).getPosition().getX();
                w[1] += alphas[i] * obstacles.get(i).getLabel() * obstacles.get(i).getPosition().getY();
            }
        }
    }

    private void extractSupportVectors(List<Obstacle> obstacles) {
        System.out.println("--- Sistem Analizi: Destek Vektörleri Bulundu ---");
        for (int i = 0; i < alphas.length; i++) {
            if (alphas[i] > 0) {
                System.out.printf("Destek Vektörü: (%.1f, %.1f) - Sınıf: %d - Önem(Alpha): %.3f\n",
                        obstacles.get(i).getPosition().getX(),
                        obstacles.get(i).getPosition().getY(),
                        obstacles.get(i).getLabel(), alphas[i]);
            }
        }
        System.out.printf("Nihai Çizgi Denklemi: (%.2f * x) + (%.2f * y) + %.2f = 0\n\n", w[0], w[1], b);
    }

    @Override
    public boolean isSafe(Point2D currentPosition) {
        return Math.abs(calculatePositionValue(currentPosition)) > 0.5;
    }

    @Override
    public String evaluatePosition(Point2D currentPosition) {
        double val = calculatePositionValue(currentPosition);
        if (val > 0) return String.format("Sol Tarafındasınız (Skor: %.2f)", val);
        else return String.format("Sağ Tarafındasınız (Skor: %.2f)", val);
    }

    private double calculatePositionValue(Point2D pos) {
        return (w[0] * pos.getX()) + (w[1] * pos.getY()) + b;
    }
}

public class AutonomousNavigator {
    private SafetyModel safetyModel;

    public AutonomousNavigator(SafetyModel model) {
        this.safetyModel = model;
    }

    public void initializeSystem(List<Obstacle> obstacles) {
        System.out.println("Sistem başlatılıyor... Engeller analiz ediliyor...");
        safetyModel.train(obstacles);
        System.out.println("Güvenlik koridoru oluşturuldu.\n");
    }

    public void checkRoute(double x, double y) {
        Point2D currentPos = new Point2D(x, y);
        System.out.println("Aracın Anlık Konumu: (" + x + ", " + y + ")");
        System.out.println("Durum: " + safetyModel.evaluatePosition(currentPos));
        System.out.println("Sürüş Güvenli mi? : " + (safetyModel.isSafe(currentPos) ? "EVET" : "DİKKAT! Sınır ihlali riski!"));
        System.out.println("-");
    }

    public static void main(String[] args) {
        List<Obstacle> sensorVerileri = new ArrayList<>();
        sensorVerileri.add(new Obstacle(2.0, 3.0, 1));
        sensorVerileri.add(new Obstacle(3.0, 3.0, 1));
        sensorVerileri.add(new Obstacle(2.0, 4.0, 1));

        sensorVerileri.add(new Obstacle(7.0, 8.0, -1));
        sensorVerileri.add(new Obstacle(8.0, 8.0, -1));
        sensorVerileri.add(new Obstacle(9.0, 7.0, -1));

        SafetyModel svmModel = new SmoSvmSolver();
        AutonomousNavigator navigator = new AutonomousNavigator(svmModel);

        navigator.initializeSystem(sensorVerileri);

        navigator.checkRoute(5.0, 5.0);
        navigator.checkRoute(3.0, 4.0);
        navigator.checkRoute(8.0, 7.0);
    }
}