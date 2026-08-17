#include "gui.hpp"

#include "image.hpp"
#include "stipple.hpp"

#include <QApplication>
#include <QCloseEvent>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QFutureWatcher>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMetaObject>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
#include <QSpinBox>
#include <QSplitter>
#include <QVBoxLayout>
#include <QWidget>
#include <QtConcurrent/QtConcurrentRun>

#include <chrono>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace stipple;

class ImagePreview final : public QLabel {
public:
    explicit ImagePreview(QWidget* parent = nullptr) : QLabel(parent) {
        setAlignment(Qt::AlignCenter);
        setMinimumSize(480, 360);
        setFrameShape(QFrame::StyledPanel);
        setStyleSheet("QLabel { background: #20242b; color: #c8ccd4; "
                      "border: 1px solid #4a505a; border-radius: 6px; }");
        showMessage("Preview hasil akan muncul di sini");
    }

    void setImage(const QImage& image) {
        image_ = image;
        setText({});
        refreshPixmap();
    }

    void showMessage(const QString& message) {
        image_ = {};
        setPixmap({});
        setText(message);
    }

protected:
    void resizeEvent(QResizeEvent* event) override {
        QLabel::resizeEvent(event);
        refreshPixmap();
    }

private:
    void refreshPixmap() {
        if (image_.isNull()) return;
        QSize target = size() - QSize(20, 20);
        setPixmap(QPixmap::fromImage(image_).scaled(
            target, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    QImage image_;
};

struct GuiSettings {
    QString inputPath;
    QString outputPath;
    std::string implName;
    int numPoints = 0;
    int maxIter = 0;
    float epsilon = 0.0f;
};

struct GuiJobResult {
    bool ok = false;
    QString error;
    QString outputPath;
    QImage image;
    RunResult run;
    int width = 0;
    int height = 0;
    int numPoints = 0;
};

class StippleWindow final : public QWidget {
public:
    StippleWindow() {
        setWindowTitle("Stipple Me This");
        setMinimumSize(920, 620);
        resize(1180, 720);

        auto* title = new QLabel("Stipple Me This");
        QFont titleFont = title->font();
        titleFont.setPointSize(18);
        titleFont.setBold(true);
        title->setFont(titleFont);

        auto* subtitle = new QLabel(
            "Weighted Lloyd stippling — pilih gambar, atur parameter, lalu lihat hasilnya.");
        subtitle->setWordWrap(true);
        subtitle->setStyleSheet("color: #596273;");

        auto* inputRow = new QWidget;
        auto* inputLayout = new QHBoxLayout(inputRow);
        inputLayout->setContentsMargins(0, 0, 0, 0);
        inputEdit_ = new QLineEdit;
        inputEdit_->setObjectName("inputPathEdit");
        inputEdit_->setPlaceholderText("Pilih gambar sumber...");
        auto* inputBrowse = new QPushButton("Pilih...");
        inputLayout->addWidget(inputEdit_, 1);
        inputLayout->addWidget(inputBrowse);

        pointsSpin_ = new QSpinBox;
        pointsSpin_->setObjectName("pointsSpin");
        pointsSpin_->setRange(1, 10000000);
        pointsSpin_->setValue(2000);
        pointsSpin_->setGroupSeparatorShown(true);

        iterationsSpin_ = new QSpinBox;
        iterationsSpin_->setObjectName("iterationsSpin");
        iterationsSpin_->setRange(1, 10000);
        iterationsSpin_->setValue(50);

        epsilonSpin_ = new QDoubleSpinBox;
        epsilonSpin_->setObjectName("epsilonSpin");
        epsilonSpin_->setRange(0.0, 1000000.0);
        epsilonSpin_->setDecimals(4);
        epsilonSpin_->setSingleStep(0.05);
        epsilonSpin_->setValue(0.10);

        implementationCombo_ = new QComboBox;
        implementationCombo_->setObjectName("implementationCombo");
        implementationCombo_->addItem("Serial", "serial");
        implementationCombo_->addItem("OpenMP (CPU paralel)", "openmp");
        implementationCombo_->addItem("SIMD (AVX2 + OpenMP)", "simd");
        implementationCombo_->addItem("CUDA (GPU NVIDIA)", "cuda");

        auto* outputRow = new QWidget;
        auto* outputLayout = new QHBoxLayout(outputRow);
        outputLayout->setContentsMargins(0, 0, 0, 0);
        outputEdit_ = new QLineEdit;
        outputEdit_->setObjectName("outputPathEdit");
        outputEdit_->setPlaceholderText("Lokasi PNG hasil...");
        auto* outputBrowse = new QPushButton("Pilih...");
        outputLayout->addWidget(outputEdit_, 1);
        outputLayout->addWidget(outputBrowse);

        auto* form = new QFormLayout;
        form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        form->setVerticalSpacing(12);
        form->addRow("Path gambar", inputRow);
        form->addRow("Jumlah titik", pointsSpin_);
        form->addRow("Maksimal iterasi", iterationsSpin_);
        form->addRow("Epsilon", epsilonSpin_);
        form->addRow("Implementasi", implementationCombo_);
        form->addRow("Path output", outputRow);

        auto* parameterGroup = new QGroupBox("Parameter komputasi");
        parameterGroup->setLayout(form);
        controls_ = parameterGroup;

        processButton_ = new QPushButton("Proses Gambar");
        processButton_->setObjectName("processButton");
        processButton_->setMinimumHeight(42);
        processButton_->setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; }"
            "QPushButton:disabled { color: #888; }");

        progressBar_ = new QProgressBar;
        progressBar_->setObjectName("progressBar");
        progressBar_->setRange(0, iterationsSpin_->value());
        progressBar_->setValue(0);
        progressBar_->setFormat("Belum diproses");

        statusLabel_ = new QLabel("Siap memproses gambar.");
        statusLabel_->setObjectName("statusLabel");
        statusLabel_->setWordWrap(true);

        resultLabel_ = new QLabel("Belum ada hasil.");
        resultLabel_->setObjectName("resultLabel");
        resultLabel_->setWordWrap(true);
        resultLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        auto* resultGroup = new QGroupBox("Informasi hasil");
        auto* resultLayout = new QVBoxLayout(resultGroup);
        resultLayout->addWidget(resultLabel_);

        auto* leftPanel = new QWidget;
        auto* leftLayout = new QVBoxLayout(leftPanel);
        leftLayout->setContentsMargins(8, 8, 8, 8);
        leftLayout->addWidget(title);
        leftLayout->addWidget(subtitle);
        leftLayout->addSpacing(8);
        leftLayout->addWidget(parameterGroup);
        leftLayout->addWidget(processButton_);
        leftLayout->addWidget(progressBar_);
        leftLayout->addWidget(statusLabel_);
        leftLayout->addWidget(resultGroup);
        leftLayout->addStretch(1);
        leftPanel->setMinimumWidth(360);
        leftPanel->setMaximumWidth(470);

        preview_ = new ImagePreview;
        preview_->setObjectName("resultPreview");
        auto* previewGroup = new QGroupBox("Preview hasil stippling");
        auto* previewLayout = new QVBoxLayout(previewGroup);
        previewLayout->addWidget(preview_);

        auto* splitter = new QSplitter(Qt::Horizontal);
        splitter->addWidget(leftPanel);
        splitter->addWidget(previewGroup);
        splitter->setStretchFactor(0, 0);
        splitter->setStretchFactor(1, 1);
        splitter->setSizes({410, 750});

        auto* root = new QHBoxLayout(this);
        root->setContentsMargins(10, 10, 10, 10);
        root->addWidget(splitter);

        watcher_ = new QFutureWatcher<GuiJobResult>(this);

        connect(inputBrowse, &QPushButton::clicked, this, [this] { chooseInput(); });
        connect(outputBrowse, &QPushButton::clicked, this, [this] { chooseOutput(); });
        connect(processButton_, &QPushButton::clicked, this, [this] { startProcessing(); });
        connect(watcher_, &QFutureWatcher<GuiJobResult>::finished,
                this, [this] { processingFinished(); });
    }

protected:
    void closeEvent(QCloseEvent* event) override {
        if (running_) {
            QMessageBox::information(
                this, "Komputasi masih berjalan",
                "Tunggu hingga pemrosesan selesai sebelum menutup aplikasi.");
            event->ignore();
            return;
        }
        QWidget::closeEvent(event);
    }

private:
    static QString defaultOutputPath(const QString& inputPath) {
        QFileInfo input(inputPath);
        QString stem = input.completeBaseName();
        if (stem.isEmpty()) stem = "hasil";
        return QDir::current().filePath("output/" + stem + "_stipple.png");
    }

    void chooseInput() {
        QString initial = inputEdit_->text().trimmed();
        if (initial.isEmpty()) initial = QDir::currentPath();
        QString path = QFileDialog::getOpenFileName(
            this, "Pilih gambar sumber", initial,
            "File gambar (*.png *.jpg *.jpeg *.bmp *.tga);;Semua file (*)");
        if (path.isEmpty()) return;
        inputEdit_->setText(QDir::toNativeSeparators(path));
        outputEdit_->setText(QDir::toNativeSeparators(defaultOutputPath(path)));
    }

    void chooseOutput() {
        QString initial = outputEdit_->text().trimmed();
        if (initial.isEmpty()) initial = defaultOutputPath(inputEdit_->text().trimmed());
        QString path = QFileDialog::getSaveFileName(
            this, "Simpan hasil stippling", initial, "PNG image (*.png)");
        if (path.isEmpty()) return;
        if (QFileInfo(path).suffix().isEmpty()) path += ".png";
        outputEdit_->setText(QDir::toNativeSeparators(path));
    }

    void startProcessing() {
        if (running_) return;

        QString inputPath = QDir::fromNativeSeparators(inputEdit_->text().trimmed());
        if (inputPath.isEmpty() || !QFileInfo(inputPath).isFile()) {
            QMessageBox::warning(this, "Input tidak valid",
                                 "Pilih file gambar yang dapat dibaca terlebih dahulu.");
            return;
        }

        QString outputPath = QDir::fromNativeSeparators(outputEdit_->text().trimmed());
        if (outputPath.isEmpty()) {
            outputPath = defaultOutputPath(inputPath);
            outputEdit_->setText(QDir::toNativeSeparators(outputPath));
        }
        if (QFileInfo(outputPath).suffix().isEmpty()) {
            outputPath += ".png";
            outputEdit_->setText(QDir::toNativeSeparators(outputPath));
        }

        if (QFileInfo(inputPath).absoluteFilePath() == QFileInfo(outputPath).absoluteFilePath()) {
            QMessageBox::warning(this, "Output tidak valid",
                                 "File output tidak boleh sama dengan gambar input.");
            return;
        }

        GuiSettings settings;
        settings.inputPath = inputPath;
        settings.outputPath = outputPath;
        settings.numPoints = pointsSpin_->value();
        settings.maxIter = iterationsSpin_->value();
        settings.epsilon = (float)epsilonSpin_->value();
        settings.implName = implementationCombo_->currentData().toString().toStdString();

        running_ = true;
        activeMaxIter_ = settings.maxIter;
        controls_->setEnabled(false);
        processButton_->setEnabled(false);
        processButton_->setText("Sedang Memproses...");
        progressBar_->setRange(0, settings.maxIter);
        progressBar_->setValue(0);
        progressBar_->setFormat("0 / %m iterasi (%p%)");
        statusLabel_->setText("Memuat gambar dan menyiapkan titik awal...");
        resultLabel_->setText("Komputasi sedang berjalan.");
        preview_->showMessage("Memproses gambar...");

        watcher_->setFuture(QtConcurrent::run([this, settings] {
            try {
                return runJob(settings);
            } catch (const std::exception& error) {
                GuiJobResult result;
                result.error = QString("Terjadi error saat pemrosesan: %1").arg(error.what());
                return result;
            } catch (...) {
                GuiJobResult result;
                result.error = "Terjadi error yang tidak diketahui saat pemrosesan.";
                return result;
            }
        }));
    }

    GuiJobResult runJob(const GuiSettings& settings) {
        GuiJobResult result;
        result.outputPath = settings.outputPath;
        result.numPoints = settings.numPoints;

        Image image;
        QByteArray inputUtf8 = settings.inputPath.toUtf8();
        if (!loadImage(inputUtf8.constData(), image, 1.4f, 0.35f)) {
            result.error = "Gambar input gagal dimuat. Periksa format dan path file.";
            return result;
        }
        if (image.w <= 0 || image.h <= 0 || image.darkArea <= 0.0) {
            result.error = "Gambar kosong atau seluruhnya putih.";
            return result;
        }

        Params params{settings.numPoints, settings.maxIter, settings.epsilon,
                      1.4f, 1.0f, 0.35f};
        std::vector<float> xs;
        std::vector<float> ys;
        initPoints(image, settings.numPoints, xs, ys, 42);

        auto started = std::chrono::steady_clock::now();
        auto onIter = [this, started, maxIter = settings.maxIter](int iter, float disp) {
            double elapsedMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            QMetaObject::invokeMethod(this, [this, iter, maxIter, disp, elapsedMs] {
                updateProgress(iter, maxIter, disp, elapsedMs);
            }, Qt::QueuedConnection);
        };

        Impl impl = implFromString(settings.implName);
        switch (impl) {
            case Impl::Serial:
                result.run = runSerial(image, params, xs, ys, onIter);
                break;
            case Impl::OpenMP:
                result.run = runOpenMP(image, params, xs, ys, onIter);
                break;
            case Impl::Simd:
                result.run = runSimd(image, params, xs, ys, onIter);
                break;
            case Impl::CUDA:
                result.run = runCUDA(image, params, xs, ys, onIter);
                break;
        }

        std::vector<uint8_t> rgb;
        renderStipple(image, xs, ys, result.run.cellWeight, rgb, 1.0f);

        QFileInfo outputInfo(settings.outputPath);
        if (!QDir().mkpath(outputInfo.absolutePath())) {
            result.error = "Direktori output tidak dapat dibuat.";
            return result;
        }
        QByteArray outputUtf8 = settings.outputPath.toUtf8();
        if (!savePNG(outputUtf8.constData(), image.w, image.h, rgb.data())) {
            result.error = "Gambar hasil gagal disimpan ke path output.";
            return result;
        }

        result.width = image.w;
        result.height = image.h;
        result.image = QImage(rgb.data(), image.w, image.h, image.w * 3,
                              QImage::Format_RGB888).copy();
        result.ok = !result.image.isNull();
        if (!result.ok) result.error = "Preview hasil gagal dibuat.";
        return result;
    }

    void updateProgress(int iter, int maxIter, float disp, double elapsedMs) {
        if (!running_) return;
        int completed = iter + 1;
        progressBar_->setMaximum(maxIter);
        progressBar_->setValue(completed);
        progressBar_->setFormat("%v / %m iterasi (%p%)");
        double etaMs = completed > 0
            ? elapsedMs / completed * (maxIter - completed)
            : 0.0;
        statusLabel_->setText(QString(
            "Iterasi %1/%2  •  max displacement %3  •  elapsed %4 s  •  ETA %5 s")
            .arg(completed)
            .arg(maxIter)
            .arg(disp, 0, 'f', 4)
            .arg(elapsedMs / 1000.0, 0, 'f', 2)
            .arg(etaMs / 1000.0, 0, 'f', 2));
    }

    void processingFinished() {
        GuiJobResult result = watcher_->result();
        running_ = false;
        controls_->setEnabled(true);
        processButton_->setEnabled(true);
        processButton_->setText("Proses Gambar");

        if (!result.ok) {
            progressBar_->setValue(0);
            progressBar_->setFormat("Gagal");
            statusLabel_->setText("Pemrosesan gagal.");
            resultLabel_->setText(result.error);
            preview_->showMessage("Preview tidak tersedia");
            QMessageBox::critical(this, "Pemrosesan gagal", result.error);
            return;
        }

        progressBar_->setValue(activeMaxIter_);
        progressBar_->setFormat("Selesai (%p%)");
        statusLabel_->setText("Pemrosesan selesai. Preview menampilkan gambar hasil.");
        resultLabel_->setText(QString(
            "Status: %1\n"
            "Ukuran: %2 × %3 px\n"
            "Titik: %4\n"
            "Iterasi: %5\n"
            "Max displacement: %6\n"
            "Waktu total: %7 ms\n"
            "Output: %8")
            .arg(result.run.converged ? "Konvergen" : "Mencapai maksimal iterasi")
            .arg(result.width)
            .arg(result.height)
            .arg(result.numPoints)
            .arg(result.run.iterations)
            .arg(result.run.maxDisp, 0, 'f', 5)
            .arg(result.run.totalMs, 0, 'f', 2)
            .arg(QDir::toNativeSeparators(result.outputPath)));
        preview_->setImage(result.image);
    }

    QLineEdit* inputEdit_ = nullptr;
    QLineEdit* outputEdit_ = nullptr;
    QSpinBox* pointsSpin_ = nullptr;
    QSpinBox* iterationsSpin_ = nullptr;
    QDoubleSpinBox* epsilonSpin_ = nullptr;
    QComboBox* implementationCombo_ = nullptr;
    QWidget* controls_ = nullptr;
    QPushButton* processButton_ = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QLabel* resultLabel_ = nullptr;
    ImagePreview* preview_ = nullptr;
    QFutureWatcher<GuiJobResult>* watcher_ = nullptr;
    bool running_ = false;
    int activeMaxIter_ = 0;
};

} // namespace

int runGui(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("Stipple Me This");
    QApplication::setOrganizationName("Stipple Me This");

    StippleWindow window;
    window.show();
    return app.exec();
}
