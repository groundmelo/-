package com.example.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.ai.ui.theme.AITheme

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AITheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    topBar = {
                        TopAppBar(
                            title = { Text("LiteRT AI Demo") },
                            colors = TopAppBarDefaults.topAppBarColors(
                                containerColor = Color(0xFF1E40AF)
                            )
                        )
                    }
                ) { innerPadding ->
                    AILayout(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun AILayout(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        PreviewSection()
        ResultSection()
        ButtonSection()
    }
}

@Composable
fun PreviewSection() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(240.dp)
            .background(Color(0xFFE5E7EB))
            .border(2.dp, Color(0xFFD1D5DB), RoundedCornerShape(8.dp)),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(
                Icons.Default.Camera,
                contentDescription = "Camera",
                modifier = Modifier.size(80.dp),
                tint = Color(0xFF9CA3AF)
            )
            Text(
                text = "Camera Preview",
                color = Color(0xFF6B7280),
                fontSize = 16.sp
            )
        }
    }
}

@Composable
fun ResultSection() {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 16.dp),
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            ResultRow(label = "Model:", value = "MobileNet")
            ResultRow(label = "Result:", value = "Cat")
            ResultRow(label = "Confidence:", value = "96.2%")
            ResultRow(label = "Time:", value = "28 ms")
        }
    }
}

@Composable
fun ResultRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF374151)
        )
        Text(
            text = value,
            fontSize = 16.sp,
            color = Color(0xFF1F2937)
        )
    }
}

@Composable
fun ButtonSection() {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            ActionButton(
                text = "拍照识别",
                icon = Icons.Default.Camera,
                backgroundColor = Color(0xFF3B82F6),
                modifier = Modifier.weight(1f)
            )
            ActionButton(
                text = "相册导入",
                icon = Icons.Default.Image,
                backgroundColor = Color(0xFF22C55E),
                modifier = Modifier.weight(1f)
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            ActionButton(
                text = "切换模型",
                icon = Icons.Default.Refresh,
                backgroundColor = Color(0xFF8B5CF6),
                modifier = Modifier.weight(1f)
            )
            ActionButton(
                text = "清空结果",
                icon = Icons.Default.Delete,
                backgroundColor = Color(0xFFEF4444),
                modifier = Modifier.weight(1f)
            )
        }
    }
}

@Composable
fun ActionButton(
    text: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    backgroundColor: Color,
    modifier: Modifier = Modifier
) {
    Button(
        onClick = {},
        modifier = modifier
            .padding(8.dp)
            .height(56.dp),
        shape = RoundedCornerShape(8.dp),
        colors = androidx.compose.material3.ButtonDefaults.buttonColors(containerColor = backgroundColor)
    ) {
        Icon(
            icon,
            contentDescription = text,
            modifier = Modifier.size(24.dp)
        )
        Text(
            text = text,
            fontSize = 16.sp,
            modifier = Modifier.padding(start = 8.dp)
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Preview(showBackground = true)
@Composable
fun AILayoutPreview() {
    AITheme {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("LiteRT AI Demo") },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = Color(0xFF1E40AF)
                    )
                )
            }
        ) { innerPadding ->
            AILayout(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .padding(16.dp)
            )
        }
    }
}
