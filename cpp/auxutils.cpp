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
int AuxUtils::s