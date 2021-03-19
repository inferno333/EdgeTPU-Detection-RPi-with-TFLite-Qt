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