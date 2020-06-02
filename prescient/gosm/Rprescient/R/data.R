#' Weather Actuals for BPA
#'
#' A dataset containing the actuals for 12/2015 -> 05/2017 They are used in the
#' getVariogramScore function to compute the hourly wind correlations.
#'
#' @format date column, unused column, actuals column \describe{
#'   \item{date}{normal datetime} \item{actuals}{actual wind generation value in mW} ... }
#' @source The BPA releases
"actuals"
