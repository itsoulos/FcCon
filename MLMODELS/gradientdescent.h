#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H
# include <CORE/problem.h>
# include <MLMODELS/linesearch.h>
# include <MLMODELS/fibonaccisearch.h>
# include <MLMODELS/goldensearch.h>
# include <MLMODELS/armijosearch.h>
# include <CORE/parameterlist.h>
# include <QString>
/**
 * @brief The GradientDescent class implements the gradient descent local optimizer.
 */
class GradientDescent
{
public:
    int maxiters;
    int iteration;
    /**
     * @brief paramList the list of parameters
     */
    ParameterList paramList;
    double eps;
    double rate;
    Data xpoint;
    double ypoint;
    bool hasInitialized;
    Problem *myProblem;
    LineSearch *lt;
    QString lineSearchMethod;//available values: none, armijo, golden, fibonacci
public:
    /**
     * @brief GradientDescent The constructor of the class.
     */
    GradientDescent(Problem *mp);
    /**
     * @brief init Initializes the parameters of the method.
     */
    virtual void init();
    /**
     * @brief step Performs a step of the optimizer.
     */
    virtual void step();
    /**
     * @brief terminated
     * @return true when the optimizer should be finished.
     */
    virtual bool terminated();
    /**
     * @brief updaterate Updates the search rate.
     */
    void    updaterate();
    /**
     * @brief updatepoint Updates the current point.
     */
    void    updatepoint();
    /**
     * @brief showDebug Displays debug information.
     */
    virtual void showDebug();
    /**
     * @brief setPoint Sets the initial point.
     * @param x
     * @param y
     */
    void    setPoint(Data &x,double &y);
    /**
     * @brief getPoint Returns the located local minimum.
     * @param x
     * @param y
     */
    void    getPoint(Data &x,double &y);
    /**
     * @brief addParam adds a new parameter
     * @param p
     */
    void        addParam(Parameter p);
    /**
     * @brief setParam adds a new parameter
     * @param name
     * @param value
     * @param help
     */
    void        setParam(QString name,QString value,QString help="");
    /**
     * @brief getParam
     * @param name
     * @return alter the parameter
     */
    Parameter   getParam(QString name);
    /**
     * @brief getParams
     * @return the parameters in json format
     */
    QJsonObject getParams();
    /**
     * @brief setParams changes the parameters
     * @param x
     */
    void        setParams(QJsonObject &x);
    /**
     * @brief init, executed before the method starts
     */
    virtual ~GradientDescent();
};

#endif // GRADIENTDESCENT_H
