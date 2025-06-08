<?php
require_once("Models/detail.php");
class DetailController
{
    var $detail_model;

    public function __construct()
    {
       $this->detail_model = new Detail();
    }
    
    public function list ()
    {   
        if(isset($_GET['id'])){
            $data = $this->detail_model->chitietbacsi($_GET['id']);
            $data_lich = $this->detail_model->kehoach($_GET['id']);
            $ngay = $this->detail_model->ngay($_GET['id']);
            // foreach ($ngay as){
            //     $gio = $this->detail_model->giotheongays($_GET['id']
            // }
            // $gio = $this->detail_model->giotheongays($_GET['id']
        }else { 
            
        }



        require_once('Views/index.php');
    }


}
