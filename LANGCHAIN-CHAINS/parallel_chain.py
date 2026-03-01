from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel

import os
load_dotenv()

model1 = ChatOpenAI(
    model="gpt-oss-120b",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt1=PromptTemplate(
    template='Generate short and simple notes from the following text \n{text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge provided notes and quiz in to single document \n notes->{notes} and quiz {quiz}',
    input_variables=['notes', 'quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {
        'notes':prompt1|model1|parser,
        'quiz':prompt2|model2|parser
    }
)

merge_chain=prompt3|model2|parser

chain=parallel_chain|merge_chain

text="""
LinearRegression
class sklearn.linear_model.LinearRegression(*, fit_intercept=True, copy_X=True, tol=1e-06, n_jobs=None, positive=False)[source]
Ordinary least squares Linear Regression.

LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

Parameters:
fit_interceptbool, default=True
Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).

copy_Xbool, default=True
If True, X will be copied; else, it may be overwritten.

tolfloat, default=1e-6
The precision of the solution (coef_) is determined by tol which specifies a different convergence criterion for the lsqr solver. tol is set as atol and btol of scipy.sparse.linalg.lsqr when fitting on sparse training data. This parameter has no effect when fitting on dense data.

Added in version 1.7.

n_jobsint, default=None
The number of jobs to use for the computation. This will only provide speedup in case of sufficiently large problems, that is if firstly n_targets > 1 and secondly X is sparse or if positive is set to True. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

positivebool, default=False
When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.

For a comparison between a linear regression model with positive constraints on the regression coefficients and a linear regression without such constraints, see Non-negative least squares.

Added in version 0.24.

Attributes:
coef_array of shape (n_features, ) or (n_targets, n_features)
Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of length n_features.

rank_int
Rank of matrix X. Only available when X is dense.

singular_array of shape (min(X, y),)
Singular values of X. Only available when X is dense.

intercept_float or array of shape (n_targets,)
Independent term in the linear model. Set to 0.0 if fit_intercept = False.

n_features_in_int
Number of features seen during fit.

Added in version 0.24.

feature_names_in_ndarray of shape (n_features_in_,)
Names of features seen during fit. Defined only when X has feature names that are all strings.

Added in version 1.0.

See also

Ridge
Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients with l2 regularization.

Lasso
The Lasso is a linear model that estimates sparse coefficients with l1 regularization.

ElasticNet
Elastic-Net is a linear regression model trained with both l1 and l2 -norm regularization of the coefficients.

Notes

From the implementation point of view, this is just plain Ordinary Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares (scipy.optimize.nnls) wrapped as a predictor object.

Examples

import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)
1.0
reg.coef_
array([1., 2.])
reg.intercept_
np.float64(3.0)
reg.predict(np.array([[3, 5]]))
array([16.])
fit(X, y, sample_weight=None)[source]
Fit linear model.

Parameters:
X{array-like, sparse matrix} of shape (n_samples, n_features)
Training data.

yarray-like of shape (n_samples,) or (n_samples, n_targets)
Target values. Will be cast to X’s dtype if necessary.

sample_weightarray-like of shape (n_samples,), default=None
Individual weights for each sample.

Added in version 0.17: parameter sample_weight support to LinearRegression.

Returns:
self
object
Fitted Estimator.
"""
result=chain.invoke({'text':text})
print(result)

chain.get_graph().print_ascii()