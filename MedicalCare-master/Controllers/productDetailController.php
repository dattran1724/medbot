<?php
require_once('./Models/drug.php');
require_once('./Models/rating.php');
class DetailController
{
  var $drug_model;
  var $rating_model;
  public function __construct()
  {
    $this->drug_model = new Drug();
    $this->rating_model = new Rating();
  }
  public function list()
  {
    $id = $_GET['drug_id'];
    // get drug
    $drug = $this->drug_model->getDrug($id);
    // get rating of drug
    $rating_drug = $this->rating_model->getProductRating($id);
    // get the avarage rating
    $avg_rating = $this->rating_model->getAvarageRating($id);
    // get comment for drug
    $comment_drug = $this->rating_model->getDrugComments($id);

    // count star rating
    $star_rating = $this->rating_model->countStarRatings($rating_drug);
    require_once('Views/index.php');
  }
}
