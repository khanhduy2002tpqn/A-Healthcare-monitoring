using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MySql.Data;
using MySql.Data.MySqlClient;
using Newtonsoft.Json;
using System.Net.Http;

namespace WindowsFormsApp2
{
    public partial class Main : Form
    {
        public Main(string username)
        {
            InitializeComponent();
            this.username = username;
        }
        TextBox glucoseTextBox;
        TextBox bloodPressureTextBox;
        TextBox skinThicknessTextBox;
        TextBox insulinTextBox;
        TextBox bmiTextBox;
        TextBox diabetesPedigreeFunctionTextBox;
        TextBox outcomeTextBox;
        TextBox nameTextBox;
        TextBox sensorTextBox;
        TextBox genderTextBox;
        TextBox ageTextBox;
        TextBox pregnanciesTextBox;

        private void Main_Load(object sender, EventArgs e)
        {
            Connect();

            GroupBox groupBoxHealthInfo = new GroupBox();
            groupBoxHealthInfo.Text = "Thông tin sức khỏe";
            groupBoxHealthInfo.Location = new Point(10, 10);
            groupBoxHealthInfo.Size = new Size(400, 300);
            Controls.Add(groupBoxHealthInfo);

            // Create a label for the Glucose field
            Label glucoseLabel = new Label();
            glucoseLabel.Text = "Glucose:";
            glucoseLabel.Location = new Point(20, 30);
            groupBoxHealthInfo.Controls.Add(glucoseLabel);

            // Create a text box for the Glucose value
            glucoseTextBox = new TextBox();
            glucoseTextBox.Location = new Point(120, 30);
            glucoseTextBox.Size = new Size(200, 20);
            groupBoxHealthInfo.Controls.Add(glucoseTextBox);

            // Create a label for the BloodPressure field
            Label bloodPressureLabel = new Label();
            bloodPressureLabel.Text = "Blood Pressure:";
            bloodPressureLabel.Location = new Point(20, 70);
            groupBoxHealthInfo.Controls.Add(bloodPressureLabel);

            // Create a text box for the BloodPressure value
            bloodPressureTextBox = new TextBox();
            bloodPressureTextBox.Location = new Point(120, 70);
            bloodPressureTextBox.Size = new Size(200, 20);
            groupBoxHealthInfo.Controls.Add(bloodPressureTextBox);

            // Create a label for the SkinThickness field
            Label skinThicknessLabel = new Label();
            skinThicknessLabel.Text = "Skin Thickness:";
            skinThicknessLabel.Location = new Point(20, 110);
            groupBoxHealthInfo.Controls.Add(skinThicknessLabel);

            // Create a text box for the SkinThickness value
            skinThicknessTextBox = new TextBox();
            skinThicknessTextBox.Location = new Point(120, 110);
            skinThicknessTextBox.Size = new Size(200, 20);
            groupBoxHealthInfo.Controls.Add(skinThicknessTextBox);

            // Create a label for the Insulin field
            Label insulinLabel = new Label();
            insulinLabel.Text = "Insulin:";
            insulinLabel.Location = new Point(20, 150);
            groupBoxHealthInfo.Controls.Add(insulinLabel);

            // Create a text box for the Insulin value
            insulinTextBox = new TextBox();
            insulinTextBox.Location = new Point(120, 150);
            insulinTextBox.Size = new Size(200, 20);
            groupBoxHealthInfo.Controls.Add(insulinTextBox);

            // Create a label for the BMI field
            Label bmiLabel = new Label();
            bmiLabel.Text = "BMI:";
            bmiLabel.Location = new Point(20, 190);
            groupBoxHealthInfo.Controls.Add(bmiLabel);

            // Create a text box for the BMI value
            bmiTextBox = new TextBox();
            bmiTextBox.Location = new Point(120, 190);
            bmiTextBox.Size = new Size(200, 20);
            groupBoxHealthInfo.Controls.Add(bmiTextBox);

            // Create a label for the DiabetesPedigreeFunction field
            Label diabetesPedigreeFunctionLabel = new Label();
            diabetesPedigreeFunctionLabel.Text = "Diabetes PF:";
            diabetesPedigreeFunctionLabel.Location = new Point(20, 230);
            groupBoxHealthInfo.Controls.Add(diabetesPedigreeFunctionLabel);

            // Create a text box for the DiabetesPedigreeFunction value
            diabetesPedigreeFunctionTextBox = new TextBox();
            //diabetesPedigreeFunctionTextBox.Text = (double.Parse(insulinTextBox.Text) * double.Parse(glucoseTextBox.Text) / 22.5 ).ToString(); 
            diabetesPedigreeFunctionTextBox.Location = new Point(120, 230);
            diabetesPedigreeFunctionTextBox.Size = new Size(200, 20);
            diabetesPedigreeFunctionTextBox.ReadOnly = true;
            groupBoxHealthInfo.Controls.Add(diabetesPedigreeFunctionTextBox);

            // Create a label for the Outcome field
            Label outcomeLabel = new Label();
            outcomeLabel.Text = "Kết quả:";
            outcomeLabel.Location = new Point(20, 270);
            groupBoxHealthInfo.Controls.Add(outcomeLabel);

            // Create a text box for the Outcome value
            outcomeTextBox = new TextBox();
            outcomeTextBox.Location = new Point(120, 270);
            outcomeTextBox.Size = new Size(200, 20);
            outcomeTextBox.ReadOnly = true;
            groupBoxHealthInfo.Controls.Add(outcomeTextBox);

            // Create a group box for user info
            GroupBox groupBoxUserInfo = new GroupBox();
            groupBoxUserInfo.Text = "Thông tin người dùng";
            groupBoxUserInfo.Location = new Point(420, 10);
            groupBoxUserInfo.Size = new Size(400, 220);
            Controls.Add(groupBoxUserInfo);

            // Create a label for the Name_client field
            Label nameLabel = new Label();
            nameLabel.Text = "Họ tên:";
            nameLabel.Location = new Point(20, 30);
            groupBoxUserInfo.Controls.Add(nameLabel);

            // Create a text box for the Name_client value
            nameTextBox = new TextBox();
            nameTextBox.Location = new Point(120, 30);
            nameTextBox.Size = new Size(200, 20);
            nameTextBox.ReadOnly = true;
            groupBoxUserInfo.Controls.Add(nameTextBox);

            // Create a label for the Sensor_id field
            Label sensorLabel = new Label();
            sensorLabel.Text = "Mã cảm biến:";
            sensorLabel.Location = new Point(20, 70);
            groupBoxUserInfo.Controls.Add(sensorLabel);

            // Create a text box for the Sensor_id value
            sensorTextBox = new TextBox();
            sensorTextBox.Location = new Point(120, 70);
            sensorTextBox.Size = new Size(200, 20);
            sensorTextBox.ReadOnly = true;
            groupBoxUserInfo.Controls.Add(sensorTextBox);

            // Create a label for the gender_client field
            Label genderLabel = new Label();
            genderLabel.Text = "Giới tính:";
            genderLabel.Location = new Point(20, 110);
            groupBoxUserInfo.Controls.Add(genderLabel);

            // Create a text box for the gender_client value
            genderTextBox = new TextBox();
            genderTextBox.Location = new Point(120, 110);
            genderTextBox.Size = new Size(200, 20);
            genderTextBox.ReadOnly = true;
            groupBoxUserInfo.Controls.Add(genderTextBox);

            // Create a label for the Pregnancies field
            Label pregnanciesLabel = new Label();
            pregnanciesLabel.Text = "Số lần mang thai:";
            pregnanciesLabel.Location = new Point(20, 150);
            groupBoxUserInfo.Controls.Add(pregnanciesLabel);

            // Create a text box for the Pregnancies value
            pregnanciesTextBox = new TextBox();
            pregnanciesTextBox.Location = new Point(120, 150);
            pregnanciesTextBox.Size = new Size(200, 20);
            pregnanciesTextBox.ReadOnly = true;
            groupBoxUserInfo.Controls.Add(pregnanciesTextBox);

            // Create a label for the Age_client field
            Label ageLabel = new Label();
            ageLabel.Text = "Tuổi:";
            ageLabel.Location = new Point(20, 190);
            groupBoxUserInfo.Controls.Add(ageLabel);

            // Create a text box for the Age_client value
            ageTextBox = new TextBox();
            ageTextBox.Location = new Point(120, 190);
            ageTextBox.Size = new Size(200, 20);
            ageTextBox.ReadOnly = true;
            groupBoxUserInfo.Controls.Add(ageTextBox);

            Getdata();
            glucose = float.Parse(glucoseTextBox.Text);
            bloodPressure = float.Parse(bloodPressureTextBox.Text);
            skinThickness = float.Parse(skinThicknessTextBox.Text);
            insulin = float.Parse(insulinTextBox.Text);
            bmi = float.Parse(bmiTextBox.Text);
            diabetesPedigreeFunction = float.Parse(diabetesPedigreeFunctionTextBox.Text);
            //.outcome = int.Parse(outcomeTextBox.Text);
            nameTextBox.Text = name;
            sensorTextBox.Text = sensor_id;

            if(gender == "0"){
                genderTextBox.Text = "Nam";
            }
            else{
                genderTextBox.Text = "Nữ";
            }
            pregnanciesTextBox.Text = pregnancies;
            ageTextBox.Text = age;
        }
        MySqlConnectionStringBuilder conn_string = new MySqlConnectionStringBuilder();
        MySqlConnection MyCon;
        MySqlCommand cmd;
        private string username;
        float glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, outcome;
        string name, sensor_id, gender, pregnancies, age;

        private async void button3_Click(object sender, EventArgs e)
        {
            var data = new {
                Pregnancies = pregnanciesTextBox.Text,
                Glucose = glucoseTextBox.Text,
                BloodPressure = bloodPressureTextBox.Text,
                SkinThickness = skinThicknessTextBox.Text,
                Insulin = insulinTextBox.Text,
                BMI = bmiTextBox.Text,
                DiabetesPedigreeFunction = diabetesPedigreeFunctionTextBox.Text,
                Age = ageTextBox.Text,
            };
            var json = JsonConvert.SerializeObject(data);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            HttpClient client = new HttpClient();
            var response = await client.PostAsync("http://192.168.120.23:8000/predict", content);
            response.EnsureSuccessStatusCode();
           // JSON {"predictions" : 1}
            string responseBody = await response.Content.ReadAsStringAsync();
            dynamic stuff = JsonConvert.DeserializeObject(responseBody);
            string predictions = stuff.predictions;
            if(predictions == "1"){
                outcomeTextBox.Text = "Có";
                outcome = 1;
            }
            else{
                outcomeTextBox.Text = "Không";
                outcome = 0;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Login login = new Login();
            login.Show();
            this.Hide();
        }

        void Connect(){
            conn_string.Server = "172.31.9.10";
            conn_string.Port = 3306;
            conn_string.UserID = "root";
            conn_string.Password = "rasp1234";
            conn_string.Database = "diabetes";
            MyCon = new MySqlConnection(conn_string.ToString());
            MyCon.Open();
        }
        private void button1_Click(object sender, EventArgs e)
        {
            // Create a connection object
            
            glucose = float.Parse(glucoseTextBox.Text);
            bloodPressure = float.Parse(bloodPressureTextBox.Text);
            skinThickness = float.Parse(skinThicknessTextBox.Text);
            insulin = float.Parse(insulinTextBox.Text);
            bmi = float.Parse(bmiTextBox.Text);
            diabetes_Pedigree_Function();
            diabetesPedigreeFunction = float.Parse(diabetesPedigreeFunctionTextBox.Text);
            // Change data from texbox from gouprboxHealthInfo to database
            string sql = "UPDATE `diabetes`.`Sensor` SET `Glucose` = '" + glucose + "', `BloodPressure` = '" + bloodPressure + "', `SkinThickness` = '" + skinThickness + "', `Insulin` = '" + insulin + "', `BMI` = '" + bmi + "', `DiabetesPedigreeFunction` = '" + diabetesPedigreeFunction + "', `Outcome` = '" + outcome + "' WHERE `Sensor`.`Sensor_id` = '" + sensor_id + "'";
            cmd = new MySqlCommand(sql, MyCon);
            cmd.ExecuteNonQuery();
            Getdata();
            MessageBox.Show("Đã lưu dữ liệu thành công!");
        }
        private void Getdata(){
            //Get data from database to textbox from groupboxUserInfo
            string sql = "SELECT * FROM `diabetes`.`user_client` WHERE `id_client` = '" + username + "'";
            cmd = new MySqlCommand(sql, MyCon);
            MySqlDataReader reader = cmd.ExecuteReader();
            //Read data from database
            if (reader.Read())
            {
                //Get data from database
                name = reader.GetString("Name_client");
                sensor_id = reader.GetString("Sensor_id");
                gender = reader.GetString("gender_client");
                pregnancies = reader.GetString("Pregnancies");
                age = reader.GetString("Age_client");
            }
            else
            {
                MessageBox.Show("Không tìm thấy dữ liệu!");
            }
            reader.Close();
            //Get data from database to textbox from groupboxHealthInfo
            sql = "SELECT * FROM `diabetes`.`Sensor` WHERE `Sensor_id` = '" + sensor_id + "'";
            cmd = new MySqlCommand(sql, MyCon);
            reader = cmd.ExecuteReader();
            //Read data from database
            if (reader.Read())
            {
                //Get data from database
                glucoseTextBox.Text = reader.GetString("Glucose");
                bloodPressureTextBox.Text = reader.GetString("BloodPressure");
                skinThicknessTextBox.Text = reader.GetString("SkinThickness");
                insulinTextBox.Text = reader.GetString("Insulin");
                bmiTextBox.Text = reader.GetString("BMI");
                diabetesPedigreeFunctionTextBox.Text = reader.GetString("DiabetesPedigreeFunction");
                //outcomeTextBox.Text = reader.GetString("Outcome");
                if (reader.GetString("Outcome") == "1")
                {
                    outcomeTextBox.Text = "Có";
                    outcome = 1;
                }
                else
                {
                    outcomeTextBox.Text = "Không";
                    outcome = 0;
                }
            }
            else
            {
                MessageBox.Show("Không tìm thấy dữ liệu!");
            }
            reader.Close();
        }
        void diabetes_Pedigree_Function()
        {
            diabetesPedigreeFunctionTextBox.Text = (insulin * glucose/22.5).ToString();
        }
    }
}
    
