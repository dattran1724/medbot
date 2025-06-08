<?php
require_once("connection.php");
// require_once(".php");
class Lichhen
{
    var $conn;
    function __construct()
    {
        $conn_obj = new connection();
        $this->conn = $conn_obj->conn;
    }
    function lenlich($id)
    {
        $query = "SELECT k.id_kh ,b.* ,s.ngay,s.gio FROM kehoach as k,status as s,bacsi as b,user as u WHERE k.id_kh = $id and k.id_status = s.id_status and k.id_bs = b.id_bs and u.userID = b.userID ;";
        require("result.php");
        return $data;
    } 
    function xemlich($id){
        $query = "SELECT * , bs.name as name_bs , bn.name as name_bn, l.trangthai as trangthai_lh FROM lichhen as l, kehoach as k , status as s , bacsi as bs , benhnhan as bn 
        WHERE k.id_kh = l.id_kh AND k.id_status = s.id_status AND l.id_bs = bs.id_bs AND l.id_bn = bn.id_bn AND l.id_bn = bn.id_bn and bn.userId = $id; " ;
        
        require("result.php");
        return $data;
    }
    function check_benhnhan($id)
    {
        $query = "SELECT * FROM benhnhan WHERE userID = $id";
        require("result.php");
        return $data;
    }
    function add_benhnhan($userID, $name, $phone2, $gioitinh, $ngaysinh, $tinh, $huyen, $xa, $diachi)
    {
        $errors = array();
        // if (empty($name)) {
        //     $errors = "Name is required." ;
        // }

        // if (!preg_match("/^[0-9]{10}$/", $phone2)) {
        //     $errors = "Invalid phone number." ;
        // }

        // if (!DateTime::createFromFormat('Y-m-d', $ngaysinh)) {
        //     $errors = "Invalid date format for ngaysinh." ;
        // }
        try {

            if (empty($errors)) {
                $query = "INSERT INTO benhnhan (userID,name,sdt,gioitinh, ngaysinh, tinh, huyen, xa, diachi) 
            VALUES (:userID,:name,:sdt,:gioitinh,:ngaysinh,:tinh,:huyen,:xa,:diachi)";
                $statement = $this->conn->prepare($query);
                $statement->bindParam(':userID', $userID);
                $statement->bindParam(':name', $name);
                $statement->bindParam(':sdt', $phone2);
                $statement->bindParam(':gioitinh', $gioitinh);
                $statement->bindParam(':ngaysinh', $ngaysinh);
                $statement->bindParam(':tinh', $tinh);
                $statement->bindParam(':huyen', $huyen);
                $statement->bindParam(':xa', $xa);
                $statement->bindParam(':diachi', $diachi);

                //  echo var_dump($statement);
                // $statement->execute();
                $success = $statement->execute();
                echo $success;
                if ($success) {
                    setcookie("msg", "Đặt lịch thành công!", time() + 2);
                    header('Location: ?act=' . 'lichhen');
                }
            } else {
                $_SESSION['errors'] = $errors;
            }
        } catch (PDOException $e) {
            setcookie("msg", "Đặt lịch thất bại!", time() + 2);
            header('Location: ?act=' . 'lichhen');
            exit;
        }


    }
    function get_benhnhan($id)
    {
        $query = "SELECT * FROM benhnhan WHERE userID = $id";

        require("result.php");

        return $data;
    }
    function get_id_bs($id)
    {
        $query = "SELECT id_bs FROM kehoach WHERE id_kh = $id";

        require("result.php");

        return $data;
    }

    function add_lichhen($id_bn, $id_bs, $id_kh, $lydo)
    {
        $errors = array();
        $trangthai = 0;
        $trangthai1 = 1;
        try {
            if (empty($errors)) {

                $query2 = "INSERT INTO lichhen (id_bn, id_bs, id_kh, lydo, trangthai) 
            VALUES (:id_bn, :id_bs, :id_kh,:lydo, :trangthai)";
                $statement = $this->conn->prepare($query2);
                $statement->bindParam(':id_bn', $id_bn);
                $statement->bindParam(':id_bs', $id_bs);
                $statement->bindParam(':id_kh', $id_kh);
                $statement->bindParam(':lydo', $lydo);
                $statement->bindParam(':trangthai', $trangthai);

                $query3 = "UPDATE kehoach SET trangthai = :trangthai1 WHERE id_kh = :id_kh";
                $statement2 = $this->conn->prepare($query3);
                $statement2->bindParam(':id_kh', $id_kh);
                $statement2->bindParam(':trangthai1', $trangthai1);

                $success = $statement->execute();
                $success2 = $statement2->execute();
                
                if ($success && $success2) {
                    return true;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } catch (PDOException $e) {
            return false;
        }
    }
}
?>