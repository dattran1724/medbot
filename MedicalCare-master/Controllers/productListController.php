<?php
require_once('./models/drug.php');
class ListController
{
  var $drug_model;
  public function __construct()
  {
    $this->drug_model = new Drug();
  }
  public function list()
  {
    $cat_id = filter_input(INPUT_GET, 'cat_id', FILTER_VALIDATE_INT);
    $page = isset($_GET['page']) ? $_GET['page'] : 1;
    $all_drug = $this->drug_model->getTotalProductByCategory($cat_id);
    $drugs = $this->drug_model->getProductByCategory($cat_id, $page);
    require_once('Views/index.php');
  }
}
