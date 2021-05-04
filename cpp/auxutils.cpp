#include "auxutils.h"
#include <QQuickItemGrabResult>
#include <QGuiApplication>
#include <QScreen>
#include <QPainter>
#include <QDir>
#include <QUuid>
#include <QJsonDocument>
#include <QList>
#include <QCryptographicHash>
#include <QThread>
#include <QQmlContext>
#include <QFile>
#include <QQuickItem>
#include <QNetworkInterface>
#include <QTimer>
#include <QStringList>

#include "colormanager.h"
#include "math.h"

double AuxUtils::dpi(QSizeF size)
{
    if (QGuiApplication::screens().count() > 0)
    {
        double dpi = QGuiApplication::screens().first()->logicalDotsPerInch();
        double lx  = 1*size.width()/QGuiApplication::screens().first()->size().width();
        double ly  = 1*size.height()/QGuiApplication::screens().first()->size().height();
        double ml  = 0.5*(lx+ly);

        return dpi * ml;
    }
    return 160;
}

// FIXME: properly implement this to be independent of the screen size and resolution
int AuxUtils::sp(int pixel, QSizeF size)
{
    //qDebug() << "Physical DPI:"  << QApplication::screens().first()->physicalDotsPerInch();
    //qDebug() << "Logical DPI:"   << QApplication::screens().first()->logicalDotsPerInch();
    //qDebug() << "Pixel ratio:"   << QApplication::screens().first()->devicePixelRatio();

    // iPhone 7: 1.5
    // iPad Mini 4: 1
    // Android: 1
    // Linux: 4
    // iPad Pro: 1
    // Raspberry Pi touch screen 7": 1

    return int(pixel * (dpi(size) / 160) * qApp->devicePixelRatio());
}

QString AuxUtils::deviceInfo()
{
    QSysInfo info;

    return  info.prettyProductName() + '\n' + '\n' +
            QString::number(QThread::idealThreadCount()) + " " + tr("cores");
}

int AuxUtils::numberThreads()
{
    return QThread::idealThreadCount();
}

QString AuxUtils::qtVersion()
{
    return qVersion();
}

QString AuxUtils::getAssetsPath()
{
    return assetsPath;
}

QImage AuxUtils::drawText(QImage image, QRectF rect, QString text, Qt::AlignmentFlag pos, Qt::GlobalColor borderColor, double borderSize, Qt::GlobalColor fontColor, QFont font)
{
    QPainter     p;
    QRectF       r = rect;
    QPainterPath path;
    QPen         pen;
    QBrush       brush;
    QStringList  lines;

    if (p.begin(&image))
    {
        // Configure font
        font.setPixelSize(AuxUtils::sp(FONT_PIXEL_SIZE_TEXT,rect.size()));
        font.setStyleHint(QFont::Times, QFont::PreferAntialias);

        // Configure pen
        pen.setWidthF(borderSize);
        pen.setStyle(Qt::SolidLine);
        pen.setColor(borderColor);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);

        // Configure brush
        brush.setStyle(Qt::SolidPattern);
        brush.setColor(fontColor);

        // Get lines
        lines = text.split('\n',QString::SkipEmptyParts);

        // Calculate text position
        QFontMetrics fm(font);
        for(int i=0;i<lines.count();i++)
        {
            // Calculate x0 and y0 positions
            int x = ((r.width()) - fm.width(lines.at(i)))/2;
            int y = pos == Qt::AlignBottom ? (r.height()) - fm.height()*(lines.count()-i) : (fm.height()*(i+1));

            // Add text to path
            path.addText(r.left()+x,r.top()+y,font,lines.at(i));
        }

        // Set pen, brush, font and draw path
        p.setRenderHints(QPainter::TextAntialiasing | QPainter::Antialiasing);
        p.setPen(pen);
        p.setBrush(brush);
        p.setFont(font);
        p.drawPath(path);
    }

    return image;
}

QImage AuxUtils::setOpacity(QImage& image, qreal opacity)
{
    QImage newImg(image.size(), QImage::Format_ARGB32);
    newImg.fill(Qt::transparent);

    QPainter painter(&newImg);
    painter.setOpacity(opacity);
    painter.drawImage(QRect(0, 0, image.width(), image.height()), image);

    return newImg;
}

QImage AuxUtils::drawMasks(QImage image, QRect rect, QStringList captions, QList<double> confidences, QList<QRectF> boxes, QList<QImage> masks,
                           double minConfidence, QMap<QString,bool> activeLabels)
{
    Q_UNUSED(captions);
    Q_UNUSED(rect);

    QPainter p;


    if (p.begin(&image))
    {
        // http://doc.qt.io/qt-5/qpainter.html#CompositionMode-enum
        p.setCompositionMode(QPainter::CompositionMode_SourceOver);

        // Draw each mask
        for(int i=0;i<masks.count();i++)
        {
            // Check min confidence value and active label
            if (confidences[i] >= minConfidence && activeLabels[captions[i]])
            {
                masks[i] = setOpacity(masks[i],MASK_OPACITY);
                p.drawImage(boxes[i].topLeft(),masks[i]);
            }
        }
    }
    return image;
}

QPointF boxCenter(QRectF rect, int offsetX=0, int offsetY=0)
{
    return QPointF(rect.left() + rect.width()*0.5 + offsetX, rect.top() + rect.height()*0.5 + offsetY);
}

QRectF pointCircle(QPointF p, double radius)
{
    return QRectF(p.x()-radius,p.y()-radius,2*radius,2*radius);
}

QRectF pointRect(QPointF p, double width, double height)
{
    return QRectF(p.x()-0.5*width,p.y()-0.5*height,width,height);
}

QPointF midPoint(QPointF a, QPointF b)
{
    return QPointF(0.5*(a.x()+b.x()),0.5*(a.y()+b.y()));
}

bool rectInside(QRectF a, QRectF b)
{
    return b.left()>=a.left() && b.top()>=a.top() && b.right()<=a.right() && b.bottom()<=a.bottom();
}

bool pointInside(QPointF p, QRectF r)
{
    return p.x()>=r.left() && p.x()<=r.right() && p.y()<=r.top() && p.y()>=r.bottom();
}

double getAngle(QPointF a, QPointF b)
{
    double angle = atan2(a.y()-b.y(),a.x()-b.x()) * 180 / M_PI;

    // Check it is not: Nan, -Inf or +Inf
    angle = angle != angle || angle > std::numeric_limits<qreal>::max() || angle < -std::numeric_limits<qreal>::max() ? 0 : angle;

    return angle;
}

QImage AuxUtils::drawBoxes(QImage image, QRect rect, QStringList captions, QList<double> confidences, QList<QRectF> boxes, double minConfidence,
                           QMap<QString, bool> activeLabels, bool rgb)
{
    Q_UNUSED(rect);

   