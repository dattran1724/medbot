<?php
require_once('./models/pharmacy.php');
require_once('./models/category.php');
require_once('./models/cart.php');
require_once('./models/rating.php');
require_once('./models/drug.php');
class PharmacyController
{
  var $pharmacy_model;
  var $category_model;
  var $cart_model;
  var $rating_model;
  public function __construct()
  {
    $this->cart_model = new Cart();
    $this->pharmacy_model = new Pharmacy();
    $this->category_model = new Category();
    $this->rating_model = new Rating();
  }
  public function list()
  {
    $categories = $this->category_model->getCategories();
    $top_ratings = $this->rating_model->getTopRatedDrugs();

    require_once('Views/index.php');
  }
}
